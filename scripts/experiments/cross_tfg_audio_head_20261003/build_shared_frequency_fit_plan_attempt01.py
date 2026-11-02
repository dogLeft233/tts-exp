#!/usr/bin/env python3
"""Freeze a fit-only, three-frontend evaluation of the shared spectral family.

This builder consumes only the eight predeclared historical fit WAVs and the
already-frozen Wav2Lip, MuseTalk, and Ditto model registries.  It emits one
paired 360-row plan and its ordered row ledger.  It does not read dev or
confirmation audio, render models, or score effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_FIT_PLAN_RAW = "e491d0bb210fe78647f4845543a88c629d37cecc09a0823c287560fbb1e71a06"
EXPECTED_W2L_PLAN_RAW = "8973187f9bc4dda91851cbf9bf53b4bca7efca08ca5db3f53e8668f061f37bb7"
EXPECTED_MUSE_PLAN_RAW = "d8414f0c41919b627bbbc24f26b97e989c0a059e82164ebe3f34b2dce34b9116"
EXPECTED_DITTO_PLAN_RAW = "3f2bebc07db7be53b0af154bd827e8408b2fdcecdc7dd478720047d1078361f2"
PROFILE_SOURCE_SHA256 = "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253"
SELECTOR_SOURCE_SHA256 = "69cee866dcc9295334c09c0d529208a9fb879b78197728b655e6ac2ccf19830b"
PROTOCOL_SOURCE_SHA256 = "43088463aee10cf49e2733905337b77c3bb0c41ed3b56781b6c14bda9ce53fe1"
TARGET_RMS = 0.0376838172675746
PROFILE_PEAK_CEILING = 0.98
FIT_SEED = 20261003
MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
METHOD_ORDER = (
    "identity",
    "rms_fixed",
    "localcompress",
    "local_r090_flatten_0.10_tilt0",
    "local_r090_accent_0.10_tilt0",
    "local_r090_flatten_0.20_tilt0",
    "local_r090_accent_0.20_tilt0",
    "local_r090_flatten_0.30_tilt0",
    "local_r090_accent_0.30_tilt0",
    "local_r090_tilt_m025",
    "local_r090_tilt_p025",
    "local_r090_flatten015_tilt_m025",
    "local_r090_accent015_tilt_p025",
    "raw_flatten020_tilt0",
    "raw_accent020_tilt0",
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def load_pinned(path: Path, expected_raw_sha256: str, label: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    actual = sha_bytes(raw)
    if actual != expected_raw_sha256:
        raise ValueError(f"{label} raw SHA mismatch: {actual}")
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object")
    return obj, actual


def _method(method_id: str, kind: str, config: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "kind": kind,
        "config": config,
        "config_sha256": sha_bytes(canonical(config)),
        "checkpoint_sha256": None,
        **extra,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    fit_plan, fit_raw = load_pinned(Path(args.fit_plan), EXPECTED_FIT_PLAN_RAW, "fit parent plan")
    w2l_plan, w2l_raw = load_pinned(Path(args.w2l_plan), EXPECTED_W2L_PLAN_RAW, "Wav2Lip parent plan")
    muse_plan, muse_raw = load_pinned(Path(args.muse_plan), EXPECTED_MUSE_PLAN_RAW, "MuseTalk parent plan")
    ditto_plan, ditto_raw = load_pinned(Path(args.ditto_plan), EXPECTED_DITTO_PLAN_RAW, "Ditto parent plan")

    fit_rows = [row for row in fit_plan.get("rows", []) if row.get("split") == "fit"]
    if len(fit_rows) != 8 or {row.get("speaker_id") for row in fit_rows} != {"S0052", "S0053", "S0054", "S0055"}:
        raise ValueError("fit roster must remain exactly the eight historical S0052-S0055 clips")
    if any(row.get("historical_exposed") is not True for row in fit_rows):
        raise ValueError("every selected fit row must be explicitly historically exposed")
    if len({row["sample_id"] for row in fit_rows}) != 8:
        raise ValueError("fit sample roster contains duplicates")

    project_root = Path(args.project_root).resolve(strict=True)
    import sys
    sys.path.insert(0, str(project_root))
    from scripts.audio_enhancement.audio_io import read_canonical_wav

    remote_audio_root = args.remote_audio_root.rstrip("/")
    audio_manifest: list[dict[str, Any]] = []
    canonical_sources: dict[str, dict[str, Any]] = {}
    for source in fit_rows:
        src = source["natural_pcm"]
        local = Path(src["transfer_local_path"]).resolve(strict=True)
        local_raw_sha = sha_file(local)
        if local_raw_sha != src["file_sha256"]:
            raise ValueError(f"fit source WAV raw SHA mismatch: {source['sample_id']}: {local_raw_sha}")
        _, canonical_info = read_canonical_wav(local)
        pcm_sha = canonical_info.pcm_float32_sha256
        if pcm_sha != src["pcm_sha256"] or canonical_info.frames != int(src["sample_count"]):
            raise ValueError(f"fit source PCM SHA/count mismatch: {source['sample_id']}")
        remote_path = f"{remote_audio_root}/{source['sample_id']}.wav"
        canonical_sources[source["sample_id"]] = {
            "path": remote_path,
            "file_sha256": local_raw_sha,
            "pcm_sha256": pcm_sha,
            "sample_count": int(src["sample_count"]),
            "sample_rate_hz": 16000,
            "channels": 1,
            "subtype": canonical_info.source_encoding,
        }
        audio_manifest.append({
            "speaker_id": source["speaker_id"],
            "sample_id": source["sample_id"],
            "local_path": str(local),
            "remote_path": remote_path,
            "bytes": local.stat().st_size,
            "file_sha256": local_raw_sha,
            "pcm_sha256": pcm_sha,
            "sample_count": int(src["sample_count"]),
            "source_archive_member": src.get("source_archive_member"),
            "source_archive_sha256": src.get("source_archive_sha256"),
        })

    model_plans = (w2l_plan, muse_plan, ditto_plan)
    models: list[dict[str, Any]] = []
    for expected_id, source_plan in zip(MODEL_ORDER, model_plans):
        model = dict(source_plan["models"][0])
        if model.get("model_id") != expected_id:
            raise ValueError(f"unexpected model order/id: {model.get('model_id')!r} != {expected_id!r}")
        if model.get("score_model_sha256") != w2l_plan["models"][0].get("score_model_sha256"):
            raise ValueError("frontends do not share the exact SyncNet checkpoint pin")
        if model.get("fps_num") != 25 or model.get("fps_den") != 1:
            raise ValueError(f"model {expected_id} does not use the frozen 25/1 clock")
        models.append(model)

    # Importing candidate_profiles also verifies the exact implementation pin.
    sys.path.insert(0, str(project_root))
    from scripts.audio_enhancement.shared_frequency_profile_v1 import (
        DEFAULT_PEAK_CEILING,
        PROFILES,
        profile_dict,
    )
    profile_file = project_root / "scripts/audio_enhancement/shared_frequency_profile_v1.py"
    if sha_file(profile_file) != PROFILE_SOURCE_SHA256:
        raise ValueError("shared profile implementation differs from the frozen source SHA")
    profile_by_id = {profile.profile_id: profile for profile in PROFILES}
    observed_profile_ids = tuple(profile_by_id)
    expected_profile_ids = METHOD_ORDER[3:]
    if observed_profile_ids != expected_profile_ids:
        raise ValueError(f"profile family order differs from frozen protocol: {observed_profile_ids}")

    methods: list[dict[str, Any]] = [
        _method("identity", "identity", {"method": "identity", "sample_rate_hz": 16000}),
        _method(
            "rms_fixed", "rms_fixed",
            {"method": "rms", "sample_rate_hz": 16000, "target_rms": TARGET_RMS, "peak_ceiling": None},
            target_rms=TARGET_RMS, target_rms_tolerance=1e-7,
        ),
        _method(
            "localcompress", "bounded_local_compression_control",
            {
                "method": "rms_then_localcompress", "sample_rate_hz": 16000,
                "pre_rms_target": TARGET_RMS, "post_rms_target": TARGET_RMS,
                "local_window_ms": 32.0, "local_ratio": 0.8,
                "local_min_gain_db": -6.0, "local_max_gain_db": 6.0,
                "envelope_floor": 1e-5,
            },
        ),
    ]
    for profile_id in expected_profile_ids:
        config = {
            "method": "shared_frequency_profile",
            "sample_rate_hz": 16000,
            "target_rms": TARGET_RMS,
            "peak_ceiling": float(DEFAULT_PEAK_CEILING),
            "profile_id": profile_id,
            "profile_spec": profile_dict(profile_id),
            "profile_source_sha256": PROFILE_SOURCE_SHA256,
        }
        methods.append(_method(profile_id, "shared_frequency_profile_candidate", config))
    if tuple(method["method_id"] for method in methods) != METHOD_ORDER:
        raise ValueError("frozen method roster/order differs from the profile selector contract")

    w2l_row0 = w2l_plan["rows"][0]
    source_face = dict(w2l_row0["source_face"])
    source_video = dict(w2l_row0["source_video"])
    if source_face.get("file_sha256") != "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6":
        raise ValueError("primary face input is not the pinned shared portrait")
    if source_video.get("file_sha256") != "c76617141188acfea5f4c122513b982997ec45ebaa35a2b6ed9f67101e9c40de":
        raise ValueError("static source video differs from the qualified shared portrait reference")

    model_by_id = {model["model_id"]: model for model in models}
    rows: list[dict[str, Any]] = []
    contexts = []
    # Context-major within model produces complete paired arms together while
    # keeping each frontend contiguous for explicit GPU scheduling.
    for model in models:
        model_id = model["model_id"]
        contexts_for_model = []
        for source in fit_rows:
            sample_id = source["sample_id"]
            speaker_id = source["speaker_id"]
            context = {
                "split": "fit", "model_id": model_id,
                "speaker_id": speaker_id, "sample_id": sample_id, "seed": FIT_SEED,
            }
            contexts_for_model.append(context)
            natural = canonical_sources[sample_id]
            for method in methods:
                method_id = method["method_id"]
                row_id = f"fit_{speaker_id}_{sample_id}_{model_id}_primary_face_static_portrait_v1_{FIT_SEED}_{method_id}"
                row = {
                    "row_id": row_id,
                    "split": "fit",
                    "sample_id": sample_id,
                    "speaker_id": speaker_id,
                    "model_id": model_id,
                    "audio_frontend": model["audio_frontend"],
                    "model_role": model.get("role", "candidate"),
                    "face_id": "primary_face_static_portrait_v1",
                    "seed": FIT_SEED,
                    "method_id": method_id,
                    "method_config_sha256": method["config_sha256"],
                    "checkpoint_sha256": method.get("checkpoint_sha256"),
                    "historical_exposed": True,
                    "natural_pcm": dict(natural),
                    "score_audio": dict(natural),
                    "source_face": dict(source_face),
                    "source_video": dict(source_video),
                    "generation_settings": {
                        "face_mode": "static_image",
                        "fps_num": 25,
                        "fps_den": 1,
                        "generator": model_id,
                        "input_audio": "transformed_natural_audio_condition_only; score_audio_is_original_natural_pcm",
                        "seed": FIT_SEED,
                    },
                }
                rows.append(row)
        contexts.extend(contexts_for_model)

    contrast_rows = [
        {"contrast_id": f"{profile_id}_minus_rms_fixed", "method_a": profile_id, "method_b": "rms_fixed", "role": "fit_only_selection"}
        for profile_id in expected_profile_ids
    ]
    protocol = json.loads(json.dumps(w2l_plan["protocol"]))
    protocol["generator"] = {
        "model_scope": "model_registry_per_frontend",
        "face_id": "primary_face_static_portrait_v1",
        "source_video_role": "shared pinned static portrait reference; each frontend uses the same source portrait input",
        "input_audio": "transformed natural float32 is generator conditioning only; original natural PCM is losslessly muxed and scored",
        "fps_num": 25,
        "fps_den": 1,
        "retiming": False,
    }
    run_id = "cross_tfg_shared_frequency_profiles_fit_20261003_attempt01"
    fit_speakers = sorted({row["speaker_id"] for row in fit_rows})
    fit_samples = sorted({row["sample_id"] for row in fit_rows})
    config_payload = {
        "fit_parent_plan_raw_sha256": fit_raw,
        "model_parent_plan_raw_sha256": {model["model_id"]: raw for model, raw in zip(models, (w2l_raw, muse_raw, ditto_raw))},
        "method_config_sha256": {method["method_id"]: method["config_sha256"] for method in methods},
        "profile_source_sha256": PROFILE_SOURCE_SHA256,
        "selector_source_sha256": SELECTOR_SOURCE_SHA256,
        "protocol_source_sha256": PROTOCOL_SOURCE_SHA256,
        "fit_seed": FIT_SEED,
        "target_rms": TARGET_RMS,
        "profile_peak_ceiling": PROFILE_PEAK_CEILING,
    }
    plan: dict[str, Any] = {
        "schema_version": "cross-tfg-audio-eval-v1",
        "run_id": run_id,
        "status": "FROZEN",
        "dataset": fit_plan["dataset"].get("name", "AISHELL/AISHELL-1"),
        "dataset_revision": fit_plan["dataset"].get("revision", "bbe295d530192a4cd41644b711c9aecd087df653"),
        "split_manifest_sha256": fit_plan["split_manifest_sha256"],
        "fit_speaker_ids": fit_speakers,
        "dev_speaker_ids": [],
        "confirm_speaker_ids": [],
        "historical_exposed_speaker_ids": fit_speakers,
        "historical_exposed_sample_ids": fit_samples,
        "historical_exposure_source": {
            "source_plan_path": str(Path(args.fit_plan).resolve()),
            "source_plan_raw_sha256": fit_raw,
            "source_roster": "same eight utterances used in prior exploratory full-context fit; all historically exposed",
        },
        "generation_seed_family": {"fit": [FIT_SEED], "dev": [], "confirm": []},
        "methods": methods,
        "models": models,
        "supports": w2l_plan["supports"],
        "protocol": protocol,
        "analysis": {
            "exploratory_fit_only": True,
            "aggregation": {
                "contexts_within_speaker": "equal_weight",
                "speakers_within_model": "equal_weight",
                "models": "equal_weight",
            },
            "bootstrap": dict(w2l_plan["analysis"]["bootstrap"]),
            "contrasts": contrast_rows,
            "summary_endpoints": [],
            "simultaneous_family_ids": [],
            "simultaneous_family_status": "NOT_APPLICABLE",
        "selection": {
                "selector_schema": "cross_tfg_shared_profile_fit_selection_v1",
                "selector_source_sha256": SELECTOR_SOURCE_SHA256,
                "protocol_source_sha256": PROTOCOL_SOURCE_SHA256,
                "reference_method_id": "rms_fixed",
                "objective": "equal speaker within each frontend, then equal frontend; Cbest primary with per-model penalties for negative Cbest, worse D0, worse Canchor, and 0.25 far-lag auxiliary contrast",
                "far_lag_offsets_abs_gte_frames": 4,
                "shortlist_count_max": 3,
                "selection_scope": "historical fit only; no development or confirmation rows/audio/effects",
            },
        },
        "contrasts": contrast_rows,
        "generation_repeat_qc": [],
        "source_rows_manifest": {
            "schema": "cross_tfg_shared_profile_fit_sources_v1",
            "fit_parent_plan_raw_sha256": fit_raw,
            "audio_inputs": audio_manifest,
        },
        "source_reference_plans": {
            "fit_parent": {"path": str(Path(args.fit_plan).resolve()), "raw_sha256": fit_raw, "payload_sha256": fit_plan.get("plan_payload_sha256")},
            "wav2lip_parent": {"path": str(Path(args.w2l_plan).resolve()), "raw_sha256": w2l_raw, "payload_sha256": w2l_plan.get("plan_sha256")},
            "musetalk_parent": {"path": str(Path(args.muse_plan).resolve()), "raw_sha256": muse_raw, "payload_sha256": muse_plan.get("plan_sha256")},
            "ditto_parent": {"path": str(Path(args.ditto_plan).resolve()), "raw_sha256": ditto_raw, "payload_sha256": ditto_plan.get("plan_sha256")},
        },
        "config_sha256": sha_bytes(canonical(config_payload)),
        "execution": {
            "stage": "historical_fit_profile_selection",
            "scope": "8 predeclared historical fit clips x 3 qualified frontends x 1 seed x 15 planned methods; no dev/confirm audio, rows, or effects",
            "planned_rows": len(rows),
            "planned_contexts_per_model": len(fit_rows),
            "profile_method_count": 12,
            "control_method_count": 3,
            "fit_seed": FIT_SEED,
            "profile_source_sha256": PROFILE_SOURCE_SHA256,
            "selector_source_sha256": SELECTOR_SOURCE_SHA256,
            "row_order": "model roster order; source fit-row order; frozen method roster order",
        },
        "fit_roster": [
            {"split": "fit", "speaker_id": row["speaker_id"], "sample_id": row["sample_id"], "seed": FIT_SEED}
            for row in fit_rows
        ],
        "rows": rows,
    }
    if len(rows) != 360 or len({row["row_id"] for row in rows}) != 360:
        raise ValueError("the frozen three-model fit cross-product must contain 360 unique rows")
    plan["plan_sha256"] = sha_bytes(canonical(plan))
    raw_plan = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    plan_path = output / "plan.json"
    rows_path = output / "planned_rows.jsonl"
    plan_path.write_bytes(raw_plan)
    rows_path.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    receipt = {
        "schema": "cross_tfg_shared_profile_fit_plan_build_receipt_v1",
        "status": "FROZEN_FIT_ONLY",
        "run_id": run_id,
        "plan_path": str(plan_path.resolve()),
        "plan_raw_sha256": sha_file(plan_path),
        "plan_payload_sha256": plan["plan_sha256"],
        "planned_rows_path": str(rows_path.resolve()),
        "planned_rows_sha256": sha_file(rows_path),
        "planned_rows": len(rows),
        "models": list(MODEL_ORDER),
        "methods": list(METHOD_ORDER),
        "contexts_per_model": len(fit_rows),
        "fit_speakers": fit_speakers,
        "confirm_rows": 0,
        "audio_inputs": audio_manifest,
        "source_plan_raw_sha256": {"fit": fit_raw, "wav2lip": w2l_raw, "musetalk": muse_raw, "ditto": ditto_raw},
        "source_sha256": {"profile": PROFILE_SOURCE_SHA256, "selector": SELECTOR_SOURCE_SHA256, "protocol": PROTOCOL_SOURCE_SHA256},
    }
    receipt_path = output / "build_receipt.json"
    receipt_path.write_bytes(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-plan", default="runs/cross_tfg_audio_head_20261003/fit_v2_fullcontext_attempt02/plan.json")
    parser.add_argument("--w2l-plan", default="runs/cross_tfg_audio_head_20261003/joint_v3_analysis_attempt03/readback_attempt01/joint_package/plan.json")
    parser.add_argument("--muse-plan", default="runs/cross_tfg_audio_head_20261003/musetalk_stageb_attempt04/frozen_plan/plan.json")
    parser.add_argument("--ditto-plan", default=".codex/scratch/cross_tfg_ditto_stageb_attempt03/packet_attempt07/plan/plan.json")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--remote-audio-root", default="/dev/shm/cross_tfg_audio_head_20261003/fit_v2_fullcontext_attempt02/input/full_audio")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    receipt = build(args)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
