"""Freeze the v4 training-only objective over the exact v2 historical fit rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


RUNTIME_SCHEMA = "cross_tfg_wav2lip_fullcontext_fit_runtime_v4_attempt01"
REMOTE_V2_PLAN_PATH = "/root/autodl-tmp/tts-research-storage-20261001/job_control/cross_tfg_audio_head_20261003_fit_v2_attempt02/plan.json"
EXPECTED_V2_PLAN_SHA256 = "e491d0bb210fe78647f4845543a88c629d37cecc09a0823c287560fbb1e71a06"
EXPECTED_V2_PLAN_PAYLOAD_SHA256 = "7200d93801ca0749b472937645d1e955e5b39c730f6a2d1a5e24d3f380f6dce9"
FIT_SOURCE_RELS = (
    "scripts/audio_enhancement/__init__.py",
    "scripts/audio_enhancement/audio_io.py",
    "scripts/audio_enhancement/baselines.py",
    "scripts/audio_enhancement/differentiable_adapter.py",
    "scripts/audio_enhancement/differentiable_native_crop.py",
    "scripts/audio_enhancement/fit_cli.py",
    "scripts/audio_enhancement/fit_cli_v4_attempt01.py",
    "scripts/audio_enhancement/heads.py",
    "scripts/audio_enhancement/overdub.py",
    "scripts/audio_enhancement/pipeline.py",
    "scripts/audio_enhancement/training.py",
    "scripts/audio_enhancement/wav2lip_fit_runtime.py",
    "scripts/audio_enhancement/wav2lip_frontend.py",
    "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v2_attempt02.py",
    "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py",
)

OBJECTIVE_WEIGHTS = {
    "relative_Cbest_improvement": 1.0,
    "far_lag_contrast_improvement": 0.25,
    "D0_nonworsening_guard": 2.0,
    "Canchor_nonworsening_guard": 1.0,
    "visual_embedding_log_norm_ratio_squared": 0.1,
    "waveform_relative_mse_to_rms_reference": 0.02,
    "mask_total_variation": 0.002,
}

OBJECTIVE = {
    "reward": "(Cbest_head-Cbest_RMS)+0.25*(far_abs4plus_head-far_abs4plus_RMS)-2*relu(D0_head-D0_RMS)^2-relu(Canchor_RMS-Canchor_head)^2",
    "zero_lag_column": 15,
    "distance_curve_reduction": "v2 FullContextSyncAdapter.sync_distance_curve; float32 query mean; all 27 queries",
    "far_lag_contrast": "mean(D_lag-D0) over lag_frames with absolute value >=4, after float32 per-lag query-mean reduction",
    "Cbest": "median_float32(curve)-min_float32(curve); first argmin is diagnostic only",
    "Canchor": "median_float32(curve)-curve_float32[zero_lag_column]",
    "fixed_clock": "source-clock q is unchanged; per-row measured minimum-MAE map selects only the Wav2Lip conditioning mel frame for output frame q",
    "norm_penalty": "mean((log(vnorm_head/vnorm_RMS))^2) over 27 query frames; both norms from unnormalized forward_lip outputs",
    "waveform_penalty": "mean((h(x)-R(x))^2)/max(mean(R(x)^2),1e-8)",
    "mask_tv": "mean(abs(diff_frequency(log_mask)))+mean(abs(diff_time(log_mask)))",
    "loss": "-reward+0.1*norm_penalty+0.02*waveform_penalty+0.002*mask_tv",
    "no_standalone_B_reward": True,
    "no_audio_retiming": True,
    "D0_guard_slack": 0.0,
    "Canchor_guard_slack": 0.0,
    "post_rms_match": "target",
}

RMS_REFERENCE = {
    "name": "bounded_RMS_reference",
    "implementation": "EnhancementStack.forward_with_mask with the candidate's identical global-RMS/rms-stage-bound/post-RMS/peak settings and use_spectral_head=False",
    "use_global_rms": True,
    "target_rms": 0.0376838172675746,
    "rms_stage_peak_ceiling": 0.99,
    "use_local_envelope": False,
    "use_spectral_head": False,
    "post_rms_match": "target",
    "peak_ceiling": 0.99,
    "legacy_fixed_rms_comparison": "also compute apply_global_rms(target_rms=0.0376838172675746, peak_ceiling=None) parity metrics per fit row; record active peak bounds and retain the development rms_fixed method unchanged",
}

CANDIDATE_AUDIO_TRANSFORM = {
    "name": "identity_initialized_spectral_head_after_bounded_RMS",
    "use_global_rms": True,
    "global_rms_target": 0.0376838172675746,
    "rms_stage_peak_ceiling": 0.99,
    "use_local_envelope": False,
    "use_spectral_head": True,
    "post_rms_match": "target",
    "peak_ceiling": 0.99,
}

ZERO_INIT_TOLERANCES = {
    "audio_max_abs": 1e-7,
    "visual_embedding_max_abs": 1e-5,
    "curve_max_abs": 1e-5,
    "reward_abs": 1e-6,
    "regularizer_abs": 1e-8,
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256_bytes(raw)


def _source_records(project_root: Path) -> list[dict[str, Any]]:
    records = []
    for rel in FIT_SOURCE_RELS:
        path = project_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"v4 fit source dependency missing: {rel}")
        records.append({
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return records


def build_plan(
    *,
    source_plan: Path,
    output_plan: Path,
    project_root: Path,
    expected_parent_sha256: str,
    baseline_cache_root: str,
    geometry_manifest_local_path: Path,
    geometry_manifest_remote_path: str,
    geometry_manifest_sha256: str,
) -> dict[str, Any]:
    raw = source_plan.read_bytes()
    parent_sha = _sha256_bytes(raw)
    if parent_sha != expected_parent_sha256.lower() or parent_sha != EXPECTED_V2_PLAN_SHA256:
        raise ValueError("v4 builder source plan raw SHA differs from the reviewed immutable v2 plan")
    base = json.loads(raw.decode("utf-8"))
    if base.get("run_id") != "cross_tfg_audio_head_fullcontext_fit_20261003_attempt02":
        raise ValueError("v4 builder accepts only the exact full-context v2 fit plan")
    if base.get("status") != "FIT_ONLY_FROZEN" or base.get("plan_payload_sha256") != EXPECTED_V2_PLAN_PAYLOAD_SHA256:
        raise ValueError("v2 source plan is not the exact frozen fit-only parent")
    if len(base.get("rows", [])) != 8 or any(row.get("split") != "fit" for row in base["rows"]):
        raise ValueError("v4 plan must preserve the exact eight fit-only v2 rows")
    if [row["row_id"] for row in base["rows"]] != base["fit_config"]["selected_rows"]:
        raise ValueError("v2 plan row order differs from its frozen selected_rows")
    if base["fit_config"].get("steps_requested") != 96 or base["fit_config"].get("fit_seed") != 20261003:
        raise ValueError("v4 plan expects the same 96-step/seed-20261003 historical fit design")
    geometry_manifest_local_path = geometry_manifest_local_path.resolve(strict=True)
    geometry_manifest_raw_sha = _sha256_file(geometry_manifest_local_path)
    if geometry_manifest_raw_sha != geometry_manifest_sha256.lower():
        raise ValueError("v4 native geometry manifest raw SHA differs from its frozen source pin")
    geometry_manifest_payload = json.loads(geometry_manifest_local_path.read_text(encoding="utf-8"))
    if geometry_manifest_payload.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1":
        raise ValueError("v4 native geometry manifest schema mismatch")
    if len(geometry_manifest_payload.get("rows", [])) != 8:
        raise ValueError("v4 native geometry manifest must contain exactly eight historical fit rows")
    if not geometry_manifest_remote_path.startswith("/root/"):
        raise ValueError("v4 native geometry manifest remote path must be an absolute project path")

    plan = copy.deepcopy(base)
    plan["schema"] = "cross_tfg_audio_head_fullcontext_fit_plan_v4"
    plan["schema_version"] = 4
    plan["run_id"] = "cross_tfg_audio_head_fullcontext_fit_20261003_v4_attempt01"
    plan["status"] = "FIT_ONLY_FROZEN"
    plan["v4_fit"] = {
        "schema": RUNTIME_SCHEMA,
        "parent_v2_plan_path": REMOTE_V2_PLAN_PATH,
        "parent_v2_plan_raw_sha256": parent_sha,
        "parent_v2_plan_payload_sha256": EXPECTED_V2_PLAN_PAYLOAD_SHA256,
        "row_ids_preserved_in_order": [row["row_id"] for row in plan["rows"]],
        "historical_fit_row_count": 8,
        "historical_fit_speakers": sorted({row["speaker_id"] for row in plan["rows"]}),
        "optimizer_steps": 96,
        "optimizer_epochs": 12,
        "fit_seed": 20261003,
        "objective": copy.deepcopy(OBJECTIVE),
        "objective_weights": copy.deepcopy(OBJECTIVE_WEIGHTS),
        "bounded_rms_reference": copy.deepcopy(RMS_REFERENCE),
        "candidate_audio_transform": copy.deepcopy(CANDIDATE_AUDIO_TRANSFORM),
        "zero_init_tolerances": copy.deepcopy(ZERO_INIT_TOLERANCES),
        "native_geometry_manifest": {
            "remote_path": geometry_manifest_remote_path,
            "bytes": geometry_manifest_local_path.stat().st_size,
            "sha256": geometry_manifest_raw_sha,
            "payload_sha256": geometry_manifest_payload.get("payload_sha256"),
        },
        "dev_audio_opened": False,
        "confirmation_audio_opened": False,
        "dev_or_confirmation_effects_read": False,
        "candidate_selected": False,
        "root_source_review": "REQUIRED_BEFORE_FIT_LAUNCH",
    }

    fit_cfg = plan["fit_config"]
    fit_cfg["epochs"] = 12
    fit_cfg["steps_requested"] = 96
    fit_cfg["fit_seed"] = 20261003
    fit_cfg["window_samples"] = None
    fit_cfg["context"] = "same complete natural PCM, center-32 Wav2Lip frames, 27 visual queries, full original SyncNet audio sequence and unpadded 31 lag support as v2"
    fit_cfg["proxy_loss"] = {
        "kind": "native_crop_bounded_RMS_delta_Cbest_plus_far_lag_guarded_v4",
        "objective": copy.deepcopy(OBJECTIVE),
        "objective_weights": copy.deepcopy(OBJECTIVE_WEIGHTS),
        "RMS_reference": copy.deepcopy(RMS_REFERENCE),
        "candidate_audio_transform": copy.deepcopy(CANDIDATE_AUDIO_TRANSFORM),
        "zero_init_tolerances": copy.deepcopy(ZERO_INIT_TOLERANCES),
        "selection_or_dev_score": False,
    }
    fit_cfg["v4_objective"] = {
        "schema": RUNTIME_SCHEMA,
        "expected_fit_rows": 8,
        "objective": copy.deepcopy(OBJECTIVE),
        "objective_weights": copy.deepcopy(OBJECTIVE_WEIGHTS),
        "rms_reference": {
            "name": "bounded_RMS_reference",
            "use_global_rms": True,
            "target_rms": RMS_REFERENCE["target_rms"],
            "rms_stage_peak_ceiling": RMS_REFERENCE["rms_stage_peak_ceiling"],
            "post_rms_match": RMS_REFERENCE["post_rms_match"],
            "peak_ceiling": RMS_REFERENCE["peak_ceiling"],
            "use_local_envelope": False,
            "use_spectral_head": False,
        },
        "candidate_audio_transform": {
            "name": CANDIDATE_AUDIO_TRANSFORM["name"],
            "use_global_rms": CANDIDATE_AUDIO_TRANSFORM["use_global_rms"],
            "global_rms_target": CANDIDATE_AUDIO_TRANSFORM["global_rms_target"],
            "rms_stage_peak_ceiling": CANDIDATE_AUDIO_TRANSFORM["rms_stage_peak_ceiling"],
            "use_local_envelope": False,
            "post_rms_match": CANDIDATE_AUDIO_TRANSFORM["post_rms_match"],
            "peak_ceiling": CANDIDATE_AUDIO_TRANSFORM["peak_ceiling"],
        },
        "one_sided_guard_slack": {"D0": 0.0, "Canchor": 0.0},
        "zero_init_tolerances": copy.deepcopy(ZERO_INIT_TOLERANCES),
    }
    plan["runtime"]["adapter_factory"] = "scripts.audio_enhancement.wav2lip_fullcontext_fit_runtime_v4_attempt01:create_runtime"
    plan["runtime"]["fit_v4"] = {
        "schema": RUNTIME_SCHEMA,
        "baseline_feature_cache_root": baseline_cache_root,
        "baseline_cache_schema": "cross_tfg_native512_rms_visual_feature_cache_v1",
        "native_geometry_manifest_path": geometry_manifest_remote_path,
        "native_geometry_manifest_sha256": geometry_manifest_raw_sha,
        "baseline_feature_artifacts": [
            "rms_reference_audio_float32",
            "rms_visual_embedding_float32",
            "rms_distance_curve_float32",
            "query_frame_index_int32",
        ],
        "materialize_all_fit_reference_features_before_first_update": True,
        "pre_optimizer_identity_head_parity_gate": True,
        "context_and_natural_audio_embedding_cache_inherited_from_v2": True,
    }
    plan["protocol"]["head_loss"] = {
        "kind": "native_crop_bounded_RMS_delta_Cbest_plus_far_lag_guarded_v4",
        "objective": copy.deepcopy(OBJECTIVE),
        "weights": copy.deepcopy(OBJECTIVE_WEIGHTS),
        "train_anchor_lag_frames": 0,
        "Cbest_used_for_training": True,
        "far_lag_set_abs_frames_at_least": 4,
        "B_reward_used_for_training": False,
        "native_geometry_manifest_sha256": geometry_manifest_raw_sha,
        "audio_source_clock_unchanged": True,
        "status_scope": "fit-only exploratory training proxy; requires fresh independent dev evaluation"
    }
    plan["protocol"]["training_audio_transform_v4"] = {
        "candidate_audio_transform": copy.deepcopy(CANDIDATE_AUDIO_TRANSFORM),
        "bounded_RMS_reference": copy.deepcopy(RMS_REFERENCE),
        "identity_initialization": "final spectral mask projection weight and bias are zero",
        "candidate_is_not_initialized_from_v2_checkpoint": True,
    }
    plan["source_revision"] = {
        "historical_parent_git_commit": str(base.get("git_commit", "")),
        "current_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip(),
        "working_tree_sources_pinned_in_bundle": True,
        "fit_package_source_files": _source_records(project_root),
        "plan_builder": {
            "path": "scripts/experiments/cross_tfg_audio_head_20261003/build_fullcontext_fit_plan_v4_attempt01.py",
            "bytes": Path(__file__).stat().st_size,
            "sha256": _sha256_file(Path(__file__)),
        },
        "current_working_tree_source_hashes_are_authoritative": True,
        "lineage_note": "The v4 runtime/CLI and native-crop helper are pinned as current source files; the v2 fit plan/runtime/checkpoint/results remain unchanged and are used only for fit-row/audio-cache provenance."
    }
    plan["plan_payload_sha256"] = _canonical_sha({key: value for key, value in plan.items() if key != "plan_payload_sha256"})
    if output_plan.exists():
        raise FileExistsError(f"refusing to overwrite v4 fit plan: {output_plan}")
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    def diff(parent: dict[str, Any], updated: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        deleted = sorted(set(parent) - set(updated))
        if deleted:
            result["__overlay_delete_keys__"] = deleted
        for key, value in updated.items():
            if key not in parent:
                result[key] = copy.deepcopy(value)
            elif isinstance(parent[key], dict) and isinstance(value, dict):
                nested = diff(parent[key], value)
                if nested:
                    result[key] = nested
            elif parent[key] != value:
                result[key] = copy.deepcopy(value)
        return result

    overlay = {
        "schema": "cross_tfg_audio_head_fit_plan_parent_overlay_v1",
        "schema_version": 1,
        "run_id": plan["run_id"],
        "status": "FIT_ONLY_FROZEN_REVIEW_PENDING",
        "parent_plan_ref": {
            "path": REMOTE_V2_PLAN_PATH,
            "raw_sha256": parent_sha,
            "payload_sha256": EXPECTED_V2_PLAN_PAYLOAD_SHA256,
        },
        "effective_plan_payload_sha256": plan["plan_payload_sha256"],
        "overrides": diff(base, plan),
        "root_source_review": "REQUIRED_BEFORE_FIT_LAUNCH",
        "dev_audio_opened": False,
        "confirmation_audio_opened": False,
        "dev_or_confirmation_effects_read": False,
        "fit_steps_performed": 0,
    }
    reconstructed = copy.deepcopy(base)

    def apply(parent_value: dict[str, Any], patch_value: dict[str, Any]) -> dict[str, Any]:
        merged_value = copy.deepcopy(parent_value)
        for delete_key in patch_value.get("__overlay_delete_keys__", []):
            merged_value.pop(delete_key, None)
        for patch_key, patch_item in patch_value.items():
            if patch_key == "__overlay_delete_keys__":
                continue
            if isinstance(merged_value.get(patch_key), dict) and isinstance(patch_item, dict):
                merged_value[patch_key] = apply(merged_value[patch_key], patch_item)
            else:
                merged_value[patch_key] = copy.deepcopy(patch_item)
        return merged_value

    reconstructed = apply(reconstructed, overlay["overrides"])
    if reconstructed != plan:
        raise AssertionError("parent overlay does not reconstruct the frozen effective v4 plan")
    if reconstructed.get("plan_payload_sha256") != _canonical_sha(
        {key: value for key, value in reconstructed.items() if key != "plan_payload_sha256"}
    ):
        raise AssertionError("reconstructed v4 effective plan payload checksum failed")
    overlay["overlay_payload_sha256"] = _canonical_sha(overlay)
    encoded = json.dumps(overlay, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if len(encoded) > 50_000:
        raise ValueError(f"compact v4 parent-overlay exceeds the 50 KB plan limit: {len(encoded)} bytes")
    output_plan.write_bytes(encoded)
    receipt = {
        "schema": "cross_tfg_audio_head_fit_v4_plan_build_receipt_v1",
        "status": "FIT_ONLY_FROZEN_REVIEW_PENDING",
        "plan_path": str(output_plan.resolve()),
        "plan_bytes": len(encoded),
        "plan_raw_sha256": _sha256_bytes(encoded),
        "overlay_payload_sha256": overlay["overlay_payload_sha256"],
        "effective_plan_payload_sha256": plan["plan_payload_sha256"],
        "parent_v2_plan_raw_sha256": parent_sha,
        "parent_v2_plan_payload_sha256": EXPECTED_V2_PLAN_PAYLOAD_SHA256,
        "row_count": len(plan["rows"]),
        "row_id_order_preserved": True,
        "optimizer_steps": 96,
        "historical_fit_only": True,
        "dev_audio_opened": False,
        "confirmation_audio_opened": False,
        "root_source_review": "REQUIRED_BEFORE_FIT_LAUNCH",
        "parent_reference": REMOTE_V2_PLAN_PATH,
    }
    receipt_path = output_plan.with_name("fit_v4_plan_build_receipt.json")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite v4 plan receipt: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--baseline-cache-root", required=True)
    parser.add_argument("--geometry-manifest-local-path", type=Path, required=True)
    parser.add_argument("--geometry-manifest-remote-path", required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        receipt = build_plan(
            source_plan=args.source_plan.resolve(strict=True),
            output_plan=args.output_plan.resolve(),
            project_root=args.project_root.resolve(strict=True),
            expected_parent_sha256=args.expected_parent_sha256,
            baseline_cache_root=args.baseline_cache_root,
            geometry_manifest_local_path=args.geometry_manifest_local_path,
            geometry_manifest_remote_path=args.geometry_manifest_remote_path,
            geometry_manifest_sha256=args.geometry_manifest_sha256,
        )
    except Exception as exc:
        parser.error(f"could not freeze v4 fit plan: {exc}")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
