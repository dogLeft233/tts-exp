#!/usr/bin/env python3
"""Bind fit40 transfer rows to the existing qualified three-frontend plans.

This emits preparation metadata only. It creates no candidate method rows,
does not launch a model, and refuses to run if any source manifest/plan pin
does not match the currently reviewed artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
FIT40_MANIFEST = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01/fit40_source_transfer_manifest.json"
ROSTER = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/v5_fit_expansion_roster.json"
FIT_PLAN = ROOT / "runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json"
OUTPUT = ROOT / "runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01/fit40_runtime_packet_metadata_attempt01.json"

EXPECTED_FIT40_MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
EXPECTED_ROSTER_SHA256 = "2fd42f097dbfb046c5bb5a2fba687c1812dc656fb06bf489974384f4eb78667a"
EXPECTED_FIT_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
EXPECTED_STAGEA_CORE_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
EXPECTED_MODEL_PARENT_RAW = {
    "wav2lip_parent": "8973187f9bc4dda91851cbf9bf53b4bca7efca08ca5db3f53e8668f061f37bb7",
    "musetalk_parent": "d8414f0c41919b627bbbc24f26b97e989c0a059e82164ebe3f34b2dce34b9116",
    "ditto_parent": "3f2bebc07db7be53b0af154bd827e8408b2fdcecdc7dd478720047d1078361f2",
}
EXPECTED_MODEL_IDS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_pinned(path: Path, expected_sha: str, label: str) -> tuple[dict[str, Any], str, int]:
    raw = path.read_bytes()
    digest = sha_bytes(raw)
    if digest != expected_sha:
        raise ValueError(f"{label} SHA mismatch: {digest}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, digest, len(raw)


def summarize_model(model: dict[str, Any]) -> dict[str, Any]:
    generator = model.get("generator") or {}
    generator_keys = (
        "implementation", "profile", "onnxruntime_profile", "torch_profile", "source_clock",
        "contract", "fps_num", "fps_den", "repo", "python", "checkpoint",
        "checkpoint_sha256", "runner_sha256", "child_script_sha256", "adapter_path",
        "adapter_sha256", "model_code_root", "model_source_root", "model_workspace",
        "musetalk_adapter_path", "musetalk_adapter_sha256",
    )
    clean_generator = {key: generator[key] for key in generator_keys if key in generator}
    result = {
        key: model[key]
        for key in (
            "model_id", "audio_frontend", "role", "model_sha256", "checkpoint_sha256",
            "code_sha256", "source_commit", "git_head", "score_model_sha256",
            "preprocessing_sha256", "generator_audio_preprocessing_sha256", "source_sha256",
        )
        if key in model
    }
    result["generator_runtime_pins"] = clean_generator
    if model.get("qualification_report"):
        result["qualification_report"] = model["qualification_report"]
    return result


def build() -> dict[str, Any]:
    manifest, manifest_sha, manifest_bytes = read_pinned(FIT40_MANIFEST, EXPECTED_FIT40_MANIFEST_SHA256, "fit40 source manifest")
    roster, roster_sha, roster_bytes = read_pinned(ROSTER, EXPECTED_ROSTER_SHA256, "fit40 roster")
    plan, plan_sha, plan_bytes = read_pinned(FIT_PLAN, EXPECTED_FIT_PLAN_SHA256, "qualified shared profile parent plan")
    if manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED":
        raise ValueError("fit40 manifest is not a no-treatment source bundle")
    if manifest.get("scope", {}).get("dev_audio_read") is not False or manifest.get("scope", {}).get("confirm_audio_read") is not False:
        raise ValueError("fit40 source manifest reports a prohibited split read")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 40 or len({row.get("sample_id") for row in rows}) != 40:
        raise ValueError("fit40 source manifest must bind exactly 40 unique endpoints")
    fit_ids = [f"S{n:04d}" for n in range(52, 72)]
    if set(manifest.get("fit_speakers", [])) != set(fit_ids):
        raise ValueError("fit40 speaker set differs from S0052-S0071")
    fold_set = roster.get("five_speaker_folds", {})
    folds = fold_set.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("fit40 roster must bind five speaker folds")
    flattened = [sid for fold in folds for sid in fold.get("held_out_speakers", [])]
    if sorted(flattened) != sorted(fit_ids) or any(len(fold.get("held_out_speakers", [])) != 4 for fold in folds):
        raise ValueError("fit40 roster fold mapping is incomplete or unbalanced")
    if set(roster["split_binding"]["fit_speaker_ids"][:20]) != set(fit_ids):
        raise ValueError("fit40 roster selection differs from frozen split")

    if plan.get("schema_version") is None or len(plan.get("models", [])) != 3:
        raise ValueError("existing shared fit plan must contain three qualified frontend models")
    model_ids = tuple(model.get("model_id") for model in plan["models"])
    if model_ids != EXPECTED_MODEL_IDS:
        raise ValueError(f"qualified model order changed: {model_ids}")
    parent_refs = plan.get("source_reference_plans")
    if not isinstance(parent_refs, dict):
        raise ValueError("shared plan is missing qualified model parent references")
    observed_parent_raw = {key: parent_refs[key].get("raw_sha256") for key in EXPECTED_MODEL_PARENT_RAW}
    if observed_parent_raw != EXPECTED_MODEL_PARENT_RAW:
        raise ValueError(f"qualified model parent raw hashes differ: {observed_parent_raw}")
    if plan.get("execution", {}).get("stagea_core_sha256") != EXPECTED_STAGEA_CORE_SHA256:
        raise ValueError("shared plan does not bind the pinned Stage-A scorer")
    core_path = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py"
    core_sha, core_bytes = sha_file(core_path)
    if core_sha != EXPECTED_STAGEA_CORE_SHA256:
        raise ValueError("actual Stage-A scorer source changed")
    evaluator_path = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py"
    lineage_path = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py"
    audio_io_path = ROOT / "scripts/audio_enhancement/audio_io.py"
    evaluator_sha, evaluator_bytes = sha_file(evaluator_path)
    lineage_sha, lineage_bytes = sha_file(lineage_path)
    audio_io_sha, audio_io_bytes = sha_file(audio_io_path)
    source_pins = manifest.get("source_implementation_pins", {})
    if source_pins.get("audio_io", {}).get("raw_sha256") != audio_io_sha:
        raise ValueError("fit40 source manifest and runtime audio_io pins differ")

    r090 = manifest.get("r090_control", {})
    if r090.get("status") != "PREPARED_NOT_FROZEN" or r090.get("config", {}).get("method_id") != "localcompress_r090":
        raise ValueError("r090 method config is not the expected prepared-only control")
    r090_cfg = r090["config"]["config"]
    if r090_cfg != {
        "method": "rms_then_localcompress",
        "sample_rate_hz": 16_000,
        "pre_rms_target": 0.0376838172675746,
        "post_rms_target": 0.0376838172675746,
        "local_window_ms": 32.0,
        "local_ratio": 0.90,
        "local_min_gain_db": -6.0,
        "local_max_gain_db": 6.0,
        "envelope_floor": 1e-5,
        "peak_ceiling": None,
    }:
        raise ValueError("r090 production config differs from the explicit frozen helper values")

    expected_control = ["identity", "rms_fixed", "localcompress_r090"]
    if manifest.get("runtime_transfer_metadata", {}).get("fit_method_roster") != expected_control:
        raise ValueError("prepared control roster differs from metadata")
    model_summaries = [summarize_model(model) for model in plan["models"]]
    row_bindings = [
        {
            key: row[key]
            for key in (
                "row_id", "speaker_id", "fold_index", "canonical_j", "sample_id",
                "source_archive_sha256", "source_archive_member", "source_member_sha256",
                "source_pcm_payload_sha256", "canonical_float32_pcm_sha256", "sample_count",
                "sample_rate_hz", "channels", "transfer_relative_path", "transfer_file_sha256",
                "transfer_file_bytes",
            )
        }
        for row in rows
    ]
    return {
        "schema": "cross_tfg_v5_fit40_runtime_packet_metadata_v1",
        "status": "PREPARED_FIT_ONLY_NOT_EXECUTABLE_UNTIL_SHARED540_AND_METHOD_FREEZE",
        "purpose": "bind the exact 40 fit-only endpoint WAVs, fixed speaker folds, prepared r090 operation, and qualified frontend/scoring references for a future remote run",
        "plan_is_executable": False,
        "fit_source_manifest": {
            "path": FIT40_MANIFEST.relative_to(ROOT).as_posix(),
            "raw_sha256": manifest_sha,
            "bytes": manifest_bytes,
            "selected_endpoints": 40,
            "fit_speakers": fit_ids,
            "selection_rule": "ADR013 canonical slots j=0 and j=15; no replacements",
            "member_manifest_raw_sha256": manifest["selection_scan"]["raw_sha256"],
        },
        "fit_folds": {
            "source_roster_path": ROSTER.relative_to(ROOT).as_posix(),
            "source_roster_raw_sha256": roster_sha,
            "source_roster_bytes": roster_bytes,
            "hash_rule": fold_set["hash_rule"],
            "folds": folds,
        },
        "candidate_method_budget": {
            "non_candidate_controls": ["identity", "rms_fixed"],
            "v5_candidate_or_ablation_slots_max": 6,
            "prepared_method_using_a_v5_slot": "localcompress_r090",
            "remaining_transition_slots_if_r090_retained": 5,
            "max_methods_per_clip_per_frontend": 8,
            "max_rows_one_seed_three_frontends": 960,
            "candidate_coefficients": "UNFROZEN",
            "method_order": "UNFROZEN",
        },
        "r090_control": r090,
        "qualified_model_reference_plan": {
            "path": FIT_PLAN.relative_to(ROOT).as_posix(),
            "raw_sha256": plan_sha,
            "bytes": plan_bytes,
            "source_reference_plans": {key: parent_refs[key] for key in EXPECTED_MODEL_PARENT_RAW},
            "models": model_summaries,
        },
        "scoring_runtime_source_pins": {
            "stagea_core": {"path": core_path.relative_to(ROOT).as_posix(), "raw_sha256": core_sha, "bytes": core_bytes},
            "syncnet_evaluator": {"path": evaluator_path.relative_to(ROOT).as_posix(), "raw_sha256": evaluator_sha, "bytes": evaluator_bytes},
            "pcm16_lineage": {"path": lineage_path.relative_to(ROOT).as_posix(), "raw_sha256": lineage_sha, "bytes": lineage_bytes},
            "audio_io": {"path": audio_io_path.relative_to(ROOT).as_posix(), "raw_sha256": audio_io_sha, "bytes": audio_io_bytes},
            "direct_score_source": "original fit source WAV PCM16 payload; retain original 16 kHz sample clock and exact direct-Pcm lineage; final candidate selection uses native scorer only after a future freeze",
        },
        "runtime_gate_and_transfer": {
            "shared540_gate": "must reach terminal and pass root interpretation before any V5 candidate method freeze or effect run",
            "observed_shared540_status_when_preparing": "RUNNING; no effect run started",
            "remote_task_namespace": None,
            "remote_output_root": None,
            "remote_python_interpreters": None,
            "gpu_admission_receipt": None,
            "transfer_status": "LOCAL_40_WAVS_READY; REMOTE_TRANSFER_NOT_STARTED",
            "future_transfer_contract": "transfer the 40 individually SHA-pinned WAV files and manifest to one fresh remote namespace; preserve every slot and failure denominator",
            "data_split_guards": {"fit_audio_read": 40, "dev_audio_read": False, "confirm_audio_read": False},
            "execution": {"model_forward": False, "render": False, "score": False, "fit_or_candidate_effect": False},
        },
        "rows": row_bindings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    result = build()
    raw = canonical_json(result) + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({
        "status": result["status"],
        "output_path": str(output_path),
        "output_bytes": len(raw),
        "output_raw_sha256": sha_bytes(raw),
        "plan_is_executable": result["plan_is_executable"],
        "source_rows": len(result["rows"]),
        "model_references": len(result["qualified_model_reference_plan"]["models"]),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
