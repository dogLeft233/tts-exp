"""Fresh two-face, equal-dose random-specificity continuation.

The source latents/directions are read from the sealed retry3 run.  Random
latents are recomputed with the corrected actual-float32 dose operation, then
all four arms are freshly rendered and scored against one fresh RAW-N audio
embedding cache per clip.  This producer writes row-level artifacts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_AUX_SITE_PACKAGES = os.environ.get("TTS_SOURCE_AXIS_AUX_SITE_PACKAGES", "")
if _AUX_SITE_PACKAGES and Path(_AUX_SITE_PACKAGES).is_dir() and _AUX_SITE_PACKAGES not in sys.path:
    sys.path.append(_AUX_SITE_PACKAGES)

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import evaluate, ops, randomdose_f32_v2, scoring, wav2lip

EXPECTED_RANDOM_SPEC_SHA256 = "679b11c8a67b18ac397d63b95900d55865b6dbf1ad7e8662bf3a5b2e915c8c79"
EXPECTED_SOURCE_PROTOCOL_SHA256 = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
EXPECTED_SOURCE_ROWS_SHA256 = "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab"
EXPECTED_SOURCE_RESULTS_SHA256 = "fd1f9ce1b23861de5eea7d87470f74e52451f1cc8bb050cbe85141a62975d3fa"
EXPECTED_CONTINUATION_PARENT_SHA256 = "d5fbfa99865b958a006f504e0dd7c6e5120ec194651aa639acad4d540528b862"
EXPECTED_RECOVERY_CODE_SEAL_SHA256 = "55c1512e35bf72563d0e01623f7e14f77aaa776dd94ee565926d9df9bea90417"
EXPECTED_INPUT_MANIFEST_SHA256 = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
EXPECTED_ANCHOR = {"k": 3, "column": 18, "official_offset": -3}
CONDITIONS = ("orth_plus", "random1", "random2", "random3")
FACE_IDS = ("primary", "secondary")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_ref(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        Path(__file__).name, "randomdose_f32_v2.py", "__init__.py", "evaluate.py",
        "ops.py", "scoring.py", "wav2lip.py",
    )
    return {name: _sha(package / name) for name in names}


def _flags() -> dict[str, bool]:
    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _set_flags() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _instrument(events: list[dict[str, Any]], active: dict[str, str | None]):
    originals = []
    for module, name in (
        (wav2lip, "cache_face_features"), (wav2lip, "decode_latents"),
        (scoring, "audio_embeddings_from_syncnet_units"),
        (scoring, "lip_embeddings_from_frames"), (evaluate, "_cross_distance_matrix"),
    ):
        original = getattr(module, name)
        originals.append((module, name, original))

        def wrapped(*args, __module=module.__name__, __name=name, __original=original, **kwargs):
            _set_flags()
            events.append({
                "operation": __name, "module": __module,
                "sample_id": active.get("sample_id"), "face_id": active.get("face_id"),
                "condition": active.get("condition"), "flags_at_call": _flags(),
                "time_monotonic": time.monotonic(),
            })
            return __original(*args, **kwargs)

        setattr(module, name, wrapped)
    return originals


def _read_source_inventory(args: argparse.Namespace, *, write_contexts: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_root = Path(args.source_run_dir).resolve(strict=True)
    lineage_paths = {
        "continuation_parent": source_root / "continuation_parent.json",
        "recovery_code_seal": source_root / "recovery_code_seal.json",
    }
    lineage_expected = {
        "continuation_parent": EXPECTED_CONTINUATION_PARENT_SHA256,
        "recovery_code_seal": EXPECTED_RECOVERY_CODE_SEAL_SHA256,
    }
    lineage = {}
    for name, path in lineage_paths.items():
        observed = _sha(path)
        if observed != lineage_expected[name]:
            raise ValueError(f"retry3 {name} lineage seal mismatch: {observed}")
        lineage[name] = {"path": str(path), "sha256": observed}
    source_protocol_path = source_root / "evaluation_protocol.json"
    source_rows_path = source_root / "evaluation_rows.jsonl"
    source_results_path = source_root / "evaluation_results.jsonl"
    source_seals = {"protocol": _sha(source_protocol_path), "rows": _sha(source_rows_path), "results": _sha(source_results_path)}
    expected = {
        "protocol": EXPECTED_SOURCE_PROTOCOL_SHA256,
        "rows": EXPECTED_SOURCE_ROWS_SHA256,
        "results": EXPECTED_SOURCE_RESULTS_SHA256,
    }
    if source_seals != expected:
        raise ValueError(f"sealed retry3 source hash mismatch: observed={source_seals}")
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_rows = _read_jsonl(source_rows_path)
    source_results = _read_jsonl(source_results_path)
    input_manifest_path = Path(args.input_manifest).resolve(strict=True)
    input_manifest_sha = _sha(input_manifest_path)
    if input_manifest_sha != EXPECTED_INPUT_MANIFEST_SHA256:
        raise ValueError(f"frozen input manifest SHA mismatch: {input_manifest_sha}")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    eval_rows = [row for row in input_manifest["records"] if row.get("split") == "evaluation"]
    if len(eval_rows) != 80 or len({row["speaker"] for row in eval_rows}) != 40:
        raise ValueError("auxiliary run requires the exact 80-clip / 40-speaker evaluation cohort")

    source_plan_by_sample = {
        row["sample_id"]: row for row in source_rows
        if row.get("family") == "source_axis" and row.get("source") == "natural"
        and row.get("face_id") == "primary" and row.get("condition") == "baseline"
    }
    results_by_key = {
        (row.get("sample_id"), row.get("face_id"), row.get("condition")): row
        for row in source_results
        if row.get("family") == "source_axis" and row.get("source") == "natural"
        and row.get("condition") in {"baseline", "orth_plus"}
    }
    if len(source_plan_by_sample) != 80:
        raise ValueError(f"retry3 source baseline plan incomplete: {len(source_plan_by_sample)}/80")
    if len(results_by_key) != 320:
        raise ValueError(f"retry3 primary/secondary baseline + orth_plus results incomplete: {len(results_by_key)}/320")

    calibration_summary_path = Path(args.calibration_summary).resolve(strict=True)
    calibration_axes_path = Path(args.calibration_axes).resolve(strict=True)
    calibration = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "calibration_complete":
        raise ValueError("calibration summary is not complete")
    observed_anchor = {
        "k": int(calibration.get("global_anchor", {}).get("selected_k", 10000)),
        "column": int(calibration.get("global_anchor", {}).get("anchor_column", -1)),
        "official_offset": int(calibration.get("global_anchor", {}).get("official_syncnet_offset", 10000)),
    }
    if observed_anchor != EXPECTED_ANCHOR:
        raise ValueError(f"calibration anchor mismatch: {observed_anchor}")
    if calibration.get("axes", {}).get("npz_sha256") != _sha(calibration_axes_path):
        raise ValueError("CAL axis archive differs from sealed summary")
    with np.load(calibration_axes_path, allow_pickle=False) as archive:
        d = np.asarray(archive["d_source"], dtype=np.float64)
        u = np.asarray(archive["u_level"], dtype=np.float64)
    if ops.array_sha256(d) != calibration["axes"]["d_sha256"] or ops.array_sha256(u) != calibration["axes"]["u_sha256"]:
        raise ValueError("CAL source/level axes differ from their sealed hashes")

    source_input_root = Path(args.input_root).resolve(strict=True)
    run_dir = Path(args.run_dir).resolve()
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for ordered_index, input_row in enumerate(eval_rows):
        sample_id = input_row["id"]
        plan = source_plan_by_sample.get(sample_id)
        if plan is None:
            raise ValueError(f"missing retry3 baseline input row for {sample_id}")
        natural_path = _resolve_ref(Path(args.input_root).resolve(), plan["natural_input_path"])
        input_sha = input_row["ref_sha256"]
        if plan.get("natural_input_sha256") != input_sha or _sha(natural_path) != input_sha:
            raise ValueError(f"natural input identity/hash mismatch for {sample_id}")
        old_base = results_by_key.get((sample_id, "primary", "baseline"))
        old_plus = results_by_key.get((sample_id, "primary", "orth_plus"))
        if not old_base or not old_plus or old_base.get("status") != "scored" or old_plus.get("status") != "scored":
            raise ValueError(f"retry3 baseline/orth_plus not scored for {sample_id}")
        context_path = _resolve_ref(source_root, old_base["context_npz"])
        context_meta_path = _resolve_ref(source_root, old_base["context_json"])
        context_sha, context_meta_sha = _sha(context_path), _sha(context_meta_path)
        if context_sha != old_base["context_npz_sha256"] or context_meta_sha != old_base["context_json_sha256"]:
            raise ValueError(f"retry3 source context hash mismatch for {sample_id}")
        if old_plus.get("context_npz_sha256") != context_sha or old_plus.get("context_json_sha256") != context_meta_sha:
            raise ValueError(f"retry3 baseline/orth_plus context bindings differ for {sample_id}")
        old_meta = json.loads(context_meta_path.read_text(encoding="utf-8"))
        with np.load(context_path, allow_pickle=False) as archive:
            z0 = np.asarray(archive["natural_z_fixed_float32"], dtype=np.float32)
            orth = np.asarray(archive["axis_z_orth_plus_float32"], dtype=np.float32)
            mask = np.asarray(archive["natural_speech_mask_fixed_uint8"], dtype=np.uint8)
            saved_d = np.asarray(archive["calibration_d_source_float64"], dtype=np.float64)
            saved_u = np.asarray(archive["calibration_u_level_float64"], dtype=np.float64)
            directions = [np.asarray(archive[f"axis_direction_random{i}_float64"], dtype=np.float64) for i in (1, 2, 3)]
        arrays_to_check = {
            "natural_z_fixed_float32": z0,
            "axis_z_orth_plus_float32": orth,
            "natural_speech_mask_fixed_uint8": mask,
            "calibration_d_source_float64": saved_d,
            "calibration_u_level_float64": saved_u,
            **{f"axis_direction_random{i}_float64": value for i, value in enumerate(directions, 1)},
        }
        for key, value in arrays_to_check.items():
            if old_meta.get("array_sha256", {}).get(key) != ops.array_sha256(value):
                raise ValueError(f"retry3 saved-array hash mismatch for {sample_id}/{key}")
        if not np.array_equal(saved_d, d) or not np.array_equal(saved_u, u):
            raise ValueError(f"retry3 context CAL axes differ from current sealed axes for {sample_id}")
        if z0.shape != orth.shape or z0.ndim != 2 or mask.shape != (len(z0),):
            raise ValueError(f"retry3 latent/mask shape mismatch for {sample_id}")
        if not np.isfinite(z0).all() or np.any(z0 < 0) or not np.array_equal(orth[mask == 0], z0[mask == 0]):
            raise ValueError(f"retry3 baseline/orth_plus values violate saved latent contract for {sample_id}")

        z_map: dict[str, np.ndarray] = {"orth_plus": orth.copy()}
        dose_details: dict[str, dict[str, Any]] = {}
        direction_hashes: dict[str, str] = {}
        dose_qc: dict[str, dict[str, Any]] = {}
        for i, direction in enumerate(directions, 1):
            condition = f"random{i}"
            candidate, detail = randomdose_f32_v2.apply_random_direction_f32_dose_v2(
                z0, mask.astype(bool), direction, saved_u, orth,
            )
            if not np.array_equal(candidate[mask == 0], z0[mask == 0]):
                raise AssertionError(f"corrected random arm changed inactive frames: {sample_id}/{condition}")
            z_map[condition] = candidate
            dose_details[condition] = detail
            direction_hashes[condition] = ops.array_sha256(direction)
            active_indices = np.flatnonzero(mask.astype(bool))
            target_norms = np.linalg.norm(
                orth[active_indices].astype(np.float64) - z0[active_indices].astype(np.float64), axis=1,
            )
            actual_norms = np.linalg.norm(
                candidate[active_indices].astype(np.float64) - z0[active_indices].astype(np.float64), axis=1,
            )
            abs_errors = np.abs(actual_norms - target_norms)
            tolerances = np.maximum(1e-6, 1e-6 * target_norms)
            tolerance_pass = abs_errors <= tolerances
            capped_by_frame = {
                int(row["frame_index"]): bool(row.get("capped_infeasible", row.get("infeasible", False)))
                for row in detail.get("frames", [])
            }
            capped_mask = np.asarray([capped_by_frame.get(int(index), False) for index in active_indices], dtype=bool)
            if len(capped_by_frame) != len(active_indices):
                raise AssertionError(f"random dose frame diagnostics incomplete for {sample_id}/{condition}")
            contract_pass = int(tolerance_pass.sum())
            tolerance_miss = int((~tolerance_pass).sum())
            capped_count = int(capped_mask.sum())
            noncapped_miss = int(((~tolerance_pass) & (~capped_mask)).sum())
            if int(detail.get("dose_contract_pass_frames", contract_pass)) != contract_pass:
                raise AssertionError(f"helper and saved float32 dose pass counts disagree for {sample_id}/{condition}")
            if int(detail.get("dose_tolerance_miss_frames", tolerance_miss)) != tolerance_miss:
                raise AssertionError(f"helper and saved float32 dose miss counts disagree for {sample_id}/{condition}")
            dose_qc[condition] = {
                "dose_contract_pass_frames": contract_pass,
                "dose_tolerance_miss_frames": tolerance_miss,
                "noncapped_tolerance_miss_frames": noncapped_miss,
                "capped_infeasible_frames": capped_count,
                "max_abs_actual_float32_dose_error": float(abs_errors.max(initial=0.0)),
                "p95_abs_actual_float32_dose_error": float(np.percentile(abs_errors, 95)) if len(abs_errors) else 0.0,
                "max_abs_tolerance_ratio": float(np.max(abs_errors / tolerances, initial=0.0)),
                "dose_contract_pass": bool(tolerance_miss == 0),
                "random_direction_sha256": direction_hashes[condition],
                "output_z_sha256": ops.array_sha256(candidate),
            }

        context_output = run_dir / "samples" / f"{sample_id}_latent_context.npz"
        context_json = context_output.with_suffix(".json")
        output_arrays = {
            "z_natural_fixed_float32": z0,
            "z_orth_plus_float32": orth,
            "natural_speech_mask_fixed_uint8": mask,
            "calibration_d_source_float64": saved_d,
            "calibration_u_level_float64": saved_u,
            **{f"axis_direction_random{i}_float64": direction for i, direction in enumerate(directions, 1)},
            **{f"z_{condition}_float32": z_map[condition] for condition in CONDITIONS},
        }
        if write_contexts:
            context_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(context_output, **output_arrays)
            out_context_sha = _sha(context_output)
        else:
            out_context_sha = None
        source_arrays_sha = {key: ops.array_sha256(value) for key, value in output_arrays.items()}
        dose_summary = {
            condition: {
                key: detail.get(key)
                for key in (
                    "active_frames", "inactive_frames", "dose_contract_pass_frames", "dose_tolerance_miss_frames",
                    "capped_infeasible_frames", "max_abs_actual_float32_dose_error", "p95_abs_actual_float32_dose_error",
                    "per_frame_abs_dose_error_max", "per_frame_abs_dose_error_mean",
                    "scale_min", "scale_median", "scale_max", "max_scale_cap",
                    "kkt_float64_residual_max_abs", "float32_level_dot_max_abs",
                    "float32_rounding_constraint_abs_bound_max", "float32_rounding_constraint_abs_max",
                    "output_z_sha256", "random_direction_sha256",
                )
            }
            for condition, detail in dose_details.items()
        }
        dose_summary = {
            condition: {**dose_summary[condition], **dose_qc[condition]}
            for condition in dose_details
        }
        artifact = {
            "ordered_eval_index": ordered_index,
            "sample_id": sample_id,
            "speaker_id": input_row["speaker"],
            "input_sha256": input_sha,
            "natural_input_path": str(natural_path),
            "source_context_npz": str(context_path),
            "source_context_npz_sha256": context_sha,
            "source_context_json": str(context_meta_path),
            "source_context_json_sha256": context_meta_sha,
            "source_arrays_sha256": source_arrays_sha,
            "random_direction_sha256": direction_hashes,
            "random_dose_diagnostics": dose_summary,
            "random_dose_qc": dose_qc,
            "source_lineage": lineage,
            "output_context_npz": str(context_output.relative_to(run_dir)) if write_contexts else f"samples/{sample_id}_latent_context.npz",
            "output_context_npz_sha256": out_context_sha,
            "output_context_json": f"samples/{sample_id}_latent_context.json",
            "output_context_json_sha256": None,
            "audio_context_npz": f"samples/{sample_id}_audio_context.npz",
            "audio_context_json": f"samples/{sample_id}_audio_context.json",
            "conditions": {},
        }
        for face_id in FACE_IDS:
            for condition in CONDITIONS:
                visual_id = f"aux_{condition}"
                row_id = f"{sample_id}__{face_id}__random_specificity__{condition}"
                dose_qc_row = artifact["random_dose_qc"].get(condition)
                rel = {
                    "latent_context_npz": artifact["output_context_npz"],
                    "context_npz": artifact["audio_context_npz"],
                    "context_json": artifact["audio_context_json"],
                    "visual_npz": f"visuals/{face_id}/{sample_id}__{visual_id}.npz",
                    "visual_json": f"visuals/{face_id}/{sample_id}__{visual_id}.json",
                    "video_path": f"videos/{face_id}/{sample_id}__{visual_id}.mkv",
                    "score_npz": f"scores/{row_id}.npz",
                }
                artifact["conditions"].setdefault(face_id, {})[condition] = {"visual_id": visual_id, **rel}
                rows.append({
                    "ordered_eval_index": ordered_index,
                    "row_id": row_id,
                    "sample_id": sample_id,
                    "speaker_id": input_row["speaker"],
                    "split": "evaluation",
                    "family": "random_specificity_aux",
                    "face_id": face_id,
                    "condition": condition,
                    "visual_id": visual_id,
                    "source": "natural",
                    "audio_level": "RAW_N",
                    "natural_input_path": str(natural_path),
                    "natural_input_sha256": input_sha,
                    "source_context_npz": str(context_path),
                    "source_context_npz_sha256": context_sha,
                    "source_context_json": str(context_meta_path),
                    "source_context_json_sha256": context_meta_sha,
                    "z_key": f"z_{condition}_float32",
                    "z_sha256": source_arrays_sha[f"z_{condition}_float32"],
                    "random_direction_sha256": direction_hashes.get(condition),
                    "random_dose_qc": dose_qc_row,
                    "dose_contract_expected_status": (
                        "scored" if dose_qc_row is None or dose_qc_row["dose_tolerance_miss_frames"] == 0
                        else "dose_contract_failed_with_score"
                    ),
                    "latent_context_npz": rel["latent_context_npz"],
                    "latent_context_npz_sha256": out_context_sha,
                    "latent_context_json": artifact["output_context_json"],
                    "latent_context_json_sha256": artifact["output_context_json_sha256"],
                    "context_npz": rel["context_npz"],
                    "context_json": rel["context_json"],
                    "visual_npz": rel["visual_npz"],
                    "visual_json": rel["visual_json"],
                    "video_path": rel["video_path"],
                    "score_crop_retention": "ephemeral; deterministic crop recipe and SHA are in visual_json",
                    "score_npz": rel["score_npz"],
                    "status": "planned",
                })
        if write_contexts:
            _json(context_json, {
                "sample_id": sample_id,
                "speaker_id": input_row["speaker"],
                "input_audio": {"path": str(natural_path), "sha256": input_sha},
                "source_retry3_context": {"path": str(context_path), "sha256": context_sha, "metadata_path": str(context_meta_path), "metadata_sha256": context_meta_sha},
                "source_arrays_sha256": source_arrays_sha,
                "random_direction_sha256": direction_hashes,
                "corrected_random_dose": dose_details,
                "output_context_npz": {"path": str(context_output.relative_to(run_dir)), "sha256": out_context_sha},
            })
            context_json_sha = _sha(context_json)
            artifact["output_context_json_sha256"] = context_json_sha
            for planned_row in rows:
                if planned_row["sample_id"] == sample_id:
                    planned_row["latent_context_json_sha256"] = context_json_sha
        artifacts.append(artifact)
    if len(artifacts) != 80 or len(rows) != 640:
        raise ValueError(f"expected 80 contexts and 640 planned rows; got {len(artifacts)} and {len(rows)}")
    source = {
        "source_run_dir": str(source_root),
        "source_protocol_sha256": source_seals["protocol"],
        "source_rows_sha256": source_seals["rows"],
        "source_results_sha256": source_seals["results"],
        "source_input_manifest_path": str(input_manifest_path),
        "source_input_manifest_sha256": input_manifest_sha,
        "source_calibration_summary_path": str(calibration_summary_path),
        "source_calibration_summary_sha256": _sha(calibration_summary_path),
        "source_calibration_axes_path": str(calibration_axes_path),
        "source_calibration_axes_sha256": _sha(calibration_axes_path),
        "source_lineage": lineage,
        "d_sha256": ops.array_sha256(d), "u_sha256": ops.array_sha256(u),
        "source_artifacts": artifacts,
    }
    return rows, artifacts, source


def _seal_protocol(args: argparse.Namespace, rows: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    spec_path = Path(args.spec).resolve(strict=True)
    if _sha(spec_path) != EXPECTED_RANDOM_SPEC_SHA256:
        raise ValueError("frozen random-specificity specification SHA mismatch")
    summary_path = Path(args.calibration_summary).resolve(strict=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    anchor = summary["global_anchor"]
    got = {"k": int(anchor["selected_k"]), "column": int(anchor["anchor_column"]), "official_offset": int(anchor["official_syncnet_offset"])}
    if got != EXPECTED_ANCHOR:
        raise ValueError(f"anchor must remain {EXPECTED_ANCHOR}; got {got}")
    row_bytes = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    row_path = Path(args.run_dir).resolve() / "evaluation_rows.jsonl"
    if row_path.exists():
        raise FileExistsError(f"refusing to overwrite row plan {row_path}")
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.write_bytes(row_bytes)
    inventory_path = Path(args.run_dir).resolve() / "source_artifact_inventory.json"
    inventory_payload = {
        "schema_version": 1,
        "source_run_lineage": source["source_lineage"],
        "source_artifacts": source["source_artifacts"],
    }
    inventory_bytes = (json.dumps(inventory_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    inventory_path.write_bytes(inventory_bytes)
    code_hashes = _code_hashes()
    faces = {
        "primary": {"path": str(Path(args.primary_face).resolve()), "sha256": _sha(Path(args.primary_face).resolve())},
        "secondary": {"path": str(Path(args.secondary_face).resolve()), "sha256": _sha(Path(args.secondary_face).resolve())},
    }
    protocol = {
        "schema_version": 1,
        "status": "frozen_pre_forward",
        "classification": "fresh corrected-dose random-specificity auxiliary run; row-level outputs only",
        "experiment_id": "confirm-random-specificity-20261001",
        "spec": {"path": str(spec_path), "sha256": _sha(spec_path)},
        "source_retry3": {
            "run_dir": source["source_run_dir"],
            "protocol_sha256": source["source_protocol_sha256"],
            "row_manifest_sha256": source["source_rows_sha256"],
            "results_sha256": source["source_results_sha256"],
            "continuation_parent": source["source_lineage"]["continuation_parent"],
            "recovery_code_seal": source["source_lineage"]["recovery_code_seal"],
            "input_manifest_sha256": source["source_input_manifest_sha256"],
            "calibration_summary_sha256": source["source_calibration_summary_sha256"],
            "calibration_axes_sha256": source["source_calibration_axes_sha256"],
            "d_sha256": source["d_sha256"], "u_sha256": source["u_sha256"],
            "context_count": 80,
        },
        "input_root": str(Path(args.input_root).resolve()),
        "faces": faces,
        "wav2lip": {"repo": str(Path(args.wav2lip_root).resolve()), "checkpoint": str(Path(args.wav2lip_checkpoint).resolve()), "checkpoint_sha256": _sha(Path(args.wav2lip_checkpoint).resolve())},
        "syncnet": {"repo": str(Path(args.syncnet_root).resolve()), "checkpoint": str(Path(args.syncnet_checkpoint).resolve()), "checkpoint_sha256": _sha(Path(args.syncnet_checkpoint).resolve())},
        "ffmpeg": {"path": str(Path(args.ffmpeg).resolve()), "sha256": _sha(Path(args.ffmpeg).resolve())},
        "calibration_anchor": EXPECTED_ANCHOR,
        "conditions": list(CONDITIONS),
        "faces_ids": list(FACE_IDS),
        "planned_samples": 80,
        "planned_speakers": 40,
        "planned_faces": 2,
        "planned_rows": 640,
        "score_audio": "one freshly computed RAW natural PCM16 SyncNet A cache shared by the four arms and both faces of each sample",
        "random_operation": {
            "operation_version": randomdose_f32_v2.OPERATION_VERSION,
            "module_sha256": code_hashes["randomdose_f32_v2.py"],
            "target": "actual float32 orth_plus displacement per active frame",
            "tolerance": "max(1e-6, 1e-6*target_norm)",
            "bounds": [0.0, 64.0], "bracket_start": 1.0, "bisections": 64,
            "inactive_frames": "exact copies",
            "failure_policy": "retain every planned row and report frame misses/capped infeasibility",
        },
        "runtime_policy": {"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True, "actual_flags_recorded_at_model_and_scoring_calls": True},
        "frontend": "official Wav2Lip generator and FFmpeg JPEG/OpenCV SyncNet video frontend; RAW PCM16 identity to float64 integer-amplitude SyncNet MFCC units",
        "support_policy": "shared V/A query support per sample and face after all four lip domains are known; retain full RAW-N audio domain for cross matrices and permutation null; full, valid, guard20, guard15 saved",
        "storage_policy": "retain generator MKV, unique z/A/V arrays and complete raw/unit cross/zero/lag matrices; crop MKV and official JPEGs are temporary, with crop/JPEG command and endpoint hashes retained for regeneration",
        "row_manifest": {"path": str(row_path), "sha256": hashlib.sha256(row_bytes).hexdigest()},
        "source_artifact_inventory": {
            "count": len(source["source_artifacts"]),
            "path": str(inventory_path),
            "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        },
        "random_dose_preflight": {
            condition: {
                "dose_contract_pass_frames": sum(item["random_dose_qc"][condition]["dose_contract_pass_frames"] for item in source["source_artifacts"]),
                "dose_tolerance_miss_frames": sum(item["random_dose_qc"][condition]["dose_tolerance_miss_frames"] for item in source["source_artifacts"]),
                "noncapped_tolerance_miss_frames": sum(item["random_dose_qc"][condition]["noncapped_tolerance_miss_frames"] for item in source["source_artifacts"]),
                "capped_infeasible_frames": sum(item["random_dose_qc"][condition]["capped_infeasible_frames"] for item in source["source_artifacts"]),
                "maximum_clip_abs_actual_float32_dose_error": max(item["random_dose_qc"][condition]["max_abs_actual_float32_dose_error"] for item in source["source_artifacts"]),
                "maximum_clip_p95_abs_actual_float32_dose_error": max(item["random_dose_qc"][condition]["p95_abs_actual_float32_dose_error"] for item in source["source_artifacts"]),
            }
            for condition in ("random1", "random2", "random3")
        },
        "code_sha256": code_hashes,
        "started_at_utc": _utc_now(),
    }
    _json(Path(args.run_dir).resolve() / "evaluation_protocol.json", protocol)
    _json(Path(args.run_dir).resolve() / "code_seal.json", {"code_sha256": code_hashes, "protocol_sha256": _sha(Path(args.run_dir).resolve() / "evaluation_protocol.json"), "row_manifest_sha256": _sha(row_path)})
    return protocol


def _read_raw_pcm(path: Path, expected_sha: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if _sha(path) != expected_sha:
        raise ValueError(f"RAW natural audio file hash mismatch: {path}")
    rate, pcm = wavfile.read(str(path))
    if rate != ops.SAMPLE_RATE or pcm.ndim != 1 or pcm.dtype != np.int16:
        raise ValueError(f"expected mono 16 kHz PCM16, got {rate}/{pcm.shape}/{pcm.dtype}")
    units = pcm.astype(np.float64)
    waveform = (pcm.astype(np.float64) / 32768.0).astype(np.float32)
    float_units, unit_meta = ops.pcm_to_syncnet_integer_units(waveform)
    if not np.array_equal(units, float_units):
        raise AssertionError("PCM16 to SyncNet float64 integer-unit identity failed")
    return pcm, units, {"path": str(path), "sha256": expected_sha, "sample_rate_hz": int(rate), "samples": int(pcm.size), "duration_s": float(pcm.size / rate), "dtype": str(pcm.dtype), "pcm_identity": unit_meta}


def _fresh_audio_cache(syncnet: Any, pcm: np.ndarray, units: np.ndarray, *, device: str, batch_size: int):
    count = len(units) // 640 - 5
    if count < 1:
        raise ValueError("RAW-N audio is too short for SyncNet windows")
    _set_flags()
    audio, audio_meta = scoring.audio_embeddings_from_syncnet_units(syncnet, units, window_count=count, device=device, batch_size=batch_size)
    mfcc = evaluate._mfcc_from_units(units)
    if ops.array_sha256(mfcc) != audio_meta["mfcc_sha256"]:
        raise AssertionError("RAW-N MFCC differs from SyncNet's scoring frontend")
    return audio, mfcc, audio_meta


def _render_visual(
    run_dir: Path, sample_id: str, face_id: str, condition: str, z: np.ndarray,
    frame: np.ndarray, face_meta: dict[str, Any], model: Any, face_cache: dict[int, list[Any]],
    syncnet: Any, args: argparse.Namespace, device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    visual_id = f"aux_{condition}"
    video_path = run_dir / "videos" / face_id / f"{sample_id}__{visual_id}.mkv"
    _set_flags()
    video_meta = wav2lip.render_video(model, z, frame, face_meta["generation_box_xyxy"], face_cache, video_path, args.ffmpeg, batch_size=args.wav2lip_batch_size, device=device)
    temp_parent = Path(args.jpeg_temp_root).resolve()
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{sample_id}_{face_id}_{condition}_", dir=temp_parent) as temp_dir:
        temporary_root = Path(temp_dir)
        crop_path = temporary_root / "score_crop.mkv"
        crop_meta = wav2lip.crop_score_video(video_path, face_meta["score_box"], crop_path, args.ffmpeg)
        jpeg_frames, jpeg_meta = scoring.extract_official_jpegs(crop_path, temporary_root / "jpeg", args.ffmpeg)
        _set_flags()
        lip = scoring.lip_embeddings_from_frames(syncnet, jpeg_frames, device=device, batch_size=args.syncnet_batch_size)
        crop_meta = {**crop_meta, "retention": "ephemeral; regenerated from retained source video", "path": None}
        jpeg_meta = {**jpeg_meta, "retention": "ephemeral; JPEGs removed after SyncNet embedding extraction", "jpeg_directory": None}
    feature_path = run_dir / "visuals" / face_id / f"{sample_id}__{visual_id}.npz"
    feature_json = feature_path.with_suffix(".json")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, lip_embeddings_float32=lip)
    meta = {
        "sample_id": sample_id, "face_id": face_id, "condition": condition, "visual_id": visual_id,
        "video": video_meta, "score_crop": crop_meta, "official_jpeg": jpeg_meta,
        "lip_embedding_shape": list(lip.shape), "lip_embedding_sha256": ops.array_sha256(lip),
        "latent_sha256": ops.array_sha256(z), "feature_npz": str(feature_path.relative_to(run_dir)),
        "feature_npz_sha256": _sha(feature_path), "metadata_path": str(feature_json.relative_to(run_dir)),
    }
    _json(feature_json, meta)
    return np.asarray(lip, dtype=np.float32), meta


def _row_unit(value: np.ndarray) -> np.ndarray:
    return evaluate._row_unit(value)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fresh auxiliary forwards")
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty auxiliary output: {run_dir}")
    for name in ("samples", "visuals", "videos", "scores", "faces"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    rows, artifacts, source = _read_source_inventory(args, write_contexts=True)
    noncapped_dose_misses = sum(
        item["random_dose_qc"][condition]["noncapped_tolerance_miss_frames"]
        for item in artifacts for condition in ("random1", "random2", "random3")
    )
    if noncapped_dose_misses:
        failure = {
            "status": "preflight_failed_noncapped_dose_miss",
            "planned_samples": 80, "planned_rows": 640,
            "noncapped_tolerance_miss_frames": noncapped_dose_misses,
            "code_sha256": _code_hashes(), "recorded_at_utc": _utc_now(),
        }
        _json(run_dir / "run_status.json", failure)
        _append_jsonl(run_dir / "evaluation_attempts.jsonl", failure)
        raise RuntimeError(f"random dose preflight found {noncapped_dose_misses} noncapped F32 tolerance misses")
    protocol = _seal_protocol(args, rows, source)
    _json(run_dir / "run_status.json", {"status": "sealed_preflight_complete", "planned_samples": 80, "planned_rows": 640, "completed_samples": 0, "failed_samples": 0})
    _json(run_dir / "evaluation_attempts.jsonl", {}) if False else None
    _append_jsonl(run_dir / "evaluation_attempts.jsonl", {"event": "run_started", "status": "running", "code_sha256": protocol["code_sha256"], "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"), "started_at_utc": protocol["started_at_utc"]})

    torch.manual_seed(20261001)
    np.random.seed(20261001)
    _set_flags()
    frontend = wav2lip.install_librosa_compatibility_shim()
    model, model_meta = wav2lip.load_model(Path(args.wav2lip_root).resolve(), args.wav2lip_checkpoint, device="cuda")
    syncnet, syncnet_meta = scoring.load_syncnet(Path(args.syncnet_root).resolve(), args.syncnet_checkpoint, device="cuda")
    events: list[dict[str, Any]] = []
    active: dict[str, str | None] = {"sample_id": None, "face_id": None, "condition": None}
    originals = _instrument(events, active)
    faces: dict[str, dict[str, Any]] = {}
    for face_id, face_arg in (("primary", args.primary_face), ("secondary", args.secondary_face)):
        active.update({"sample_id": None, "face_id": face_id, "condition": "shared_face_cache"})
        path = Path(face_arg).resolve()
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"cannot read {face_id} face image: {path}")
        flags_before = _flags()
        face_meta = wav2lip.detect_static_face(frame, Path(args.wav2lip_root).resolve(), device="cuda")
        flags_after_detection = _flags()
        _set_flags()
        face_meta.update({"path": str(path), "sha256": _sha(path), "width": int(frame.shape[1]), "height": int(frame.shape[0])})
        face_cache = wav2lip.cache_face_features(model, frame, face_meta["generation_box_xyxy"], range(1, args.wav2lip_batch_size + 1), device="cuda")
        faces[face_id] = {"frame": frame, "meta": face_meta, "cache": face_cache, "flags_before_detection": flags_before, "flags_after_detection": flags_after_detection, "flags_after_override": _flags()}
        _json(run_dir / "faces" / f"{face_id}.json", {"face_id": face_id, "face": face_meta, "flags_before_detection": flags_before, "flags_after_detection": flags_after_detection, "flags_after_override": _flags(), "model_checkpoint_sha256": _sha(Path(args.wav2lip_checkpoint).resolve()), "frontend": frontend})
    completed_samples = 0
    failed_samples: list[dict[str, Any]] = []
    result_path = run_dir / "evaluation_results.jsonl"
    try:
        for artifact in artifacts:
            sample_id = artifact["sample_id"]
            sample_started = time.perf_counter()
            latent_path = run_dir / artifact["output_context_npz"]
            latent_json_path = latent_path.with_suffix(".json")
            try:
                with np.load(latent_path, allow_pickle=False) as archive:
                    latent_arrays = {key: archive[key] for key in archive.files}
                z_by_condition = {condition: np.asarray(latent_arrays[f"z_{condition}_float32"], dtype=np.float32) for condition in CONDITIONS}
                mask = np.asarray(latent_arrays["natural_speech_mask_fixed_uint8"], dtype=np.uint8)
                pcm, units, audio_input_meta = _read_raw_pcm(Path(artifact["natural_input_path"]), artifact["input_sha256"])
                active.update({"sample_id": sample_id, "face_id": None, "condition": "shared_A_RAW_N"})
                audio, mfcc, audio_forward_meta = _fresh_audio_cache(syncnet, pcm, units, device="cuda", batch_size=args.syncnet_batch_size)
                audio_domain = len(audio)
                per_face_results: list[dict[str, Any]] = []
                face_context: dict[str, Any] = {}
                for face_id in FACE_IDS:
                    face = faces[face_id]
                    visual_cache: dict[str, np.ndarray] = {}
                    visual_meta: dict[str, dict[str, Any]] = {}
                    for condition in CONDITIONS:
                        active.update({"sample_id": sample_id, "face_id": face_id, "condition": condition})
                        lip, meta = _render_visual(run_dir, sample_id, face_id, condition, z_by_condition[condition], face["frame"], face["meta"], model, face["cache"], syncnet, args, "cuda")
                        if len(lip) < 1:
                            raise ValueError(f"no lip embedding rows for {sample_id}/{face_id}/{condition}")
                        visual_cache[condition] = lip
                        visual_meta[condition] = meta
                    query_n = min(audio_domain, *(len(value) for value in visual_cache.values()))
                    if query_n < 1:
                        raise ValueError(f"empty common V/A support for {sample_id}/{face_id}")
                    supports = evaluate._support_indices(query_n)
                    face_context[face_id] = {
                        "common_query_windows": int(query_n), "audio_embedding_rows": int(audio_domain),
                        "support_indices": {key: value.tolist() for key, value in supports.items()},
                        "support_sha256": {
                            key: ops.array_sha256(np.asarray(value, dtype=np.int32))
                            for key, value in supports.items()
                        },
                        "visuals": {condition: visual_meta[condition] for condition in CONDITIONS},
                    }
                    for condition in CONDITIONS:
                        active.update({"sample_id": sample_id, "face_id": face_id, "condition": condition})
                        visual = visual_cache[condition]
                        visual_unit = _row_unit(visual)
                        audio_unit = _row_unit(audio)
                        _set_flags()
                        cross_raw, zero_raw = evaluate._cross_distance_matrix(visual, audio, "cuda")
                        matrix_raw = evaluate._lag_matrix_from_cross(cross_raw, zero_raw, query_n, audio_domain, vshift=15)
                        _set_flags()
                        cross_unit, zero_unit = evaluate._cross_distance_matrix(visual_unit, audio_unit, "cuda")
                        matrix_unit = evaluate._lag_matrix_from_cross(cross_unit, zero_unit, query_n, audio_domain, vshift=15)
                        row_plan = next(row for row in rows if row["sample_id"] == sample_id and row["face_id"] == face_id and row["condition"] == condition)
                        score_path = run_dir / row_plan["score_npz"]
                        score_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            score_path,
                            cross_distance_raw_float32=cross_raw,
                            cross_distance_to_zero_raw_float32=zero_raw,
                            distance_matrix_raw_float32=matrix_raw,
                            cross_distance_unit_float32=cross_unit,
                            cross_distance_to_zero_unit_float32=zero_unit,
                            distance_matrix_unit_float32=matrix_unit,
                            common_support_full_int32=supports["full"],
                            common_support_valid_int32=supports["valid"],
                            common_support_guard20_int32=supports["guard20"],
                            common_support_guard15_int32=supports["guard15"],
                            lag_columns_int32=np.arange(31, dtype=np.int32),
                            lag_k_int32=np.arange(-15, 16, dtype=np.int32),
                        )
                        random_dose_qc = artifact["random_dose_qc"].get(condition)
                        dose_contract_failed = bool(random_dose_qc and random_dose_qc["dose_tolerance_miss_frames"] > 0)
                        result = {
                            **row_plan,
                            "status": "dose_contract_failed_with_score" if dose_contract_failed else "scored",
                            "score_computed": True,
                            "recorded_at_utc": _utc_now(),
                            "visual_npz_sha256": _sha(run_dir / row_plan["visual_npz"]),
                            "visual_json_sha256": _sha(run_dir / row_plan["visual_json"]),
                            "visual_embedding_ref": {"path": row_plan["visual_npz"], "key": "lip_embeddings_float32"},
                            "audio_embedding_ref": {"path": row_plan["context_npz"], "key": "natural_audio_embeddings_raw_float32"},
                            "video_sha256": visual_meta[condition]["video"]["sha256"],
                            "score_npz_sha256": _sha(score_path),
                            "common_support_n": int(query_n), "audio_row_domain_n": int(audio_domain),
                            "visual_query_domain_n": int(len(visual)),
                            "common_support_sha256": face_context[face_id]["support_sha256"],
                            "anchor": {"k": 3, "column": 18, "official_syncnet_offset": -3},
                            "raw_support_metrics": evaluate._metric_supports(matrix_raw, 18),
                            "unit_support_metrics": evaluate._metric_supports(matrix_unit, 18),
                            "audio_embedding_sha256": ops.array_sha256(audio),
                            "visual_embedding_sha256": ops.array_sha256(visual),
                            "latent_sha256": ops.array_sha256(z_by_condition[condition]),
                            "random_dose_summary": artifact["random_dose_diagnostics"].get(condition),
                            "random_dose_qc": random_dose_qc,
                            "flags_at_forward": sorted({json.dumps(event["flags_at_call"], sort_keys=True) for event in events if event.get("sample_id") == sample_id and event.get("face_id") == face_id and event.get("condition") == condition}),
                        }
                        per_face_results.append(result)

                # The one per-sample context is common to both faces and points to
                # all four fresh V caches through the separate visual NPZs.
                context_payload = {
                    "natural_pcm16": pcm,
                    "natural_syncnet_units_raw_float64": units,
                    "natural_mfcc_raw_float64": mfcc,
                    "natural_audio_embeddings_raw_float32": audio,
                }
                for face_id in FACE_IDS:
                    for name, values in face_context[face_id]["support_indices"].items():
                        context_payload[f"{face_id}_common_support_{name}_int32"] = np.asarray(values, dtype=np.int32)
                audio_context_path = run_dir / artifact["audio_context_npz"]
                audio_context_json_path = run_dir / artifact["audio_context_json"]
                audio_context_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(audio_context_path, **context_payload)
                context_sha = _sha(audio_context_path)
                audio_context_meta = {
                    "sample_id": sample_id,
                    "speaker_id": artifact["speaker_id"],
                    "latent_context_npz": {"path": str(latent_path.relative_to(run_dir)), "sha256": _sha(latent_path)},
                    "raw_audio_cache": {"input": audio_input_meta, "forward": audio_forward_meta, "mfcc_sha256": ops.array_sha256(mfcc), "embedding_sha256": ops.array_sha256(audio), "embedding_rows": int(audio_domain)},
                    "faces": face_context,
                    "audio_context_npz": {"path": artifact["audio_context_npz"], "sha256": context_sha},
                }
                _json(audio_context_json_path, audio_context_meta)
                context_json_sha = _sha(audio_context_json_path)
                for result in per_face_results:
                    result["context_npz"] = artifact["audio_context_npz"]
                    result["context_npz_sha256"] = context_sha
                    result["context_json"] = artifact["audio_context_json"]
                    result["context_json_sha256"] = context_json_sha
                    result["latent_context_npz"] = artifact["output_context_npz"]
                    result["latent_context_npz_sha256"] = _sha(latent_path)
                    _append_jsonl(result_path, result)
                _append_jsonl(run_dir / "evaluation_attempts.jsonl", {
                    "sample_id": sample_id, "speaker_id": artifact["speaker_id"], "status": "complete",
                    "score_rows_written": len(per_face_results),
                    "dose_contract_failed_rows": sum(row["status"] == "dose_contract_failed_with_score" for row in per_face_results),
                    "elapsed_seconds": time.perf_counter() - sample_started,
                    "context_npz_sha256": context_sha, "recorded_at_utc": _utc_now(),
                })
                completed_samples += 1
                _json(run_dir / "run_status.json", {"status": "running", "planned_samples": 80, "planned_rows": 640, "completed_samples": completed_samples, "failed_samples": len(failed_samples), "last_sample_id": sample_id})
                print(json.dumps({"event": "sample_complete", "sample_id": sample_id, "completed_samples": completed_samples, "rows_written": len(per_face_results), "elapsed_seconds": time.perf_counter() - sample_started}), flush=True)
            except Exception as exc:
                failure = {"sample_id": sample_id, "error_type": type(exc).__name__, "error": str(exc)[:2000], "traceback_tail": traceback.format_exc()[-4000:], "recorded_at_utc": _utc_now()}
                failed_samples.append(failure)
                for row in rows:
                    if row["sample_id"] == sample_id:
                        _append_jsonl(result_path, {**row, "status": "failed", **failure})
                _append_jsonl(run_dir / "evaluation_attempts.jsonl", {"status": "failed", **failure})
                _json(run_dir / "run_status.json", {"status": "running", "planned_samples": 80, "planned_rows": 640, "completed_samples": completed_samples, "failed_samples": len(failed_samples), "last_sample_id": sample_id})
                print(json.dumps({"event": "sample_failed", **failure}, ensure_ascii=False), flush=True)
    finally:
        for module, name, original in originals:
            setattr(module, name, original)

    result_rows = _read_jsonl(result_path)
    final = {
        "status": "complete_with_failures" if failed_samples else "complete",
        "classification": protocol["classification"],
        "planned_samples": 80, "completed_samples": completed_samples,
        "failed_samples": len(failed_samples), "planned_rows": 640,
        "completed_rows": sum(row.get("status") == "scored" for row in result_rows),
        "score_artifacts_written": sum(bool(row.get("score_computed")) for row in result_rows),
        "dose_contract_failed_rows": sum(row.get("status") == "dose_contract_failed_with_score" for row in result_rows),
        "failed_rows": sum(row.get("status") in {"failed", "dose_contract_failed_with_score"} for row in result_rows),
        "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"),
        "row_manifest_sha256": _sha(run_dir / "evaluation_rows.jsonl"),
        "results_sha256": _sha(result_path),
        "code_sha256": protocol["code_sha256"],
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": torch.cuda.get_device_name(), "numpy": str(np.__version__), "opencv": str(cv2.__version__), "frontend_compatibility": frontend, "wav2lip_model": model_meta, "syncnet_model": syncnet_meta},
        "forward_flag_values": sorted({json.dumps(event["flags_at_call"], sort_keys=True) for event in events}),
        "forward_events": events,
        "face_flags": {face_id: {key: value for key, value in face.items() if key.startswith("flags_")} for face_id, face in faces.items()},
        "finished_at_utc": _utc_now(),
    }
    _json(run_dir / "run_status.json", final)
    _json(run_dir / "all_files_sha256.json", {
        str(path.relative_to(run_dir)): _sha(path)
        for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "all_files_sha256.json"
    })
    return final


def _materialize_contexts_only(args: argparse.Namespace) -> dict[str, Any]:
    """Create and independently hash every non-neural context before GPU use."""
    output = Path(args.run_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"context materialization target must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows, artifacts, source = _read_source_inventory(args, write_contexts=True)
    failures: list[dict[str, str]] = []
    inventory_rows = []
    for artifact in artifacts:
        sample_id = artifact["sample_id"]
        npz_path = output / artifact["output_context_npz"]
        json_path = output / artifact["output_context_json"]
        if not npz_path.is_file() or not json_path.is_file():
            failures.append({"sample_id": sample_id, "reason": "context_file_missing"})
            continue
        npz_sha, json_sha = _sha(npz_path), _sha(json_path)
        if npz_sha != artifact["output_context_npz_sha256"]:
            failures.append({"sample_id": sample_id, "reason": "context_npz_sha_mismatch"})
        if json_sha != artifact["output_context_json_sha256"]:
            failures.append({"sample_id": sample_id, "reason": "context_json_sha_mismatch"})
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("output_context_npz", {}).get("sha256") != npz_sha:
            failures.append({"sample_id": sample_id, "reason": "json_npz_binding_mismatch"})
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if set(arrays) != set(artifact["source_arrays_sha256"]):
            failures.append({"sample_id": sample_id, "reason": "npz_array_key_mismatch"})
        for key, value in arrays.items():
            if ops.array_sha256(value) != artifact["source_arrays_sha256"].get(key):
                failures.append({"sample_id": sample_id, "reason": f"array_sha_mismatch:{key}"})
        planned = [row for row in rows if row["sample_id"] == sample_id]
        if len(planned) != 8 or any(row.get("latent_context_npz_sha256") != npz_sha or row.get("latent_context_json_sha256") != json_sha for row in planned):
            failures.append({"sample_id": sample_id, "reason": "planned_row_context_binding_mismatch"})
        inventory_rows.append({
            "sample_id": sample_id,
            "context_npz": artifact["output_context_npz"],
            "context_npz_sha256": npz_sha,
            "context_json": artifact["output_context_json"],
            "context_json_sha256": json_sha,
            "array_count": len(arrays),
            "array_sha256": artifact["source_arrays_sha256"],
        })
    inventory = {
        "schema_version": 1,
        "classification": "non-neural corrected-dose context materialization inventory",
        "source_run_lineage": source["source_lineage"],
        "artifacts": artifacts,
    }
    _json(output / "source_artifact_inventory.json", inventory)
    report = {
        "schema_version": 1,
        "status": "materialization_pass" if not failures and len(artifacts) == 80 and len(rows) == 640 else "materialization_failed",
        "classification": "non-neural source/input/dose context verification; no model loaded and no outcomes read",
        "context_count": len(artifacts),
        "planned_row_count": len(rows),
        "verified_context_count": len(inventory_rows),
        "verified_array_count": sum(row["array_count"] for row in inventory_rows),
        "integrity_failure_count": len(failures),
        "integrity_failures": failures,
        "source_artifact_inventory_path": "source_artifact_inventory.json",
        "source_artifact_inventory_sha256": _sha(output / "source_artifact_inventory.json"),
        "contexts": inventory_rows,
    }
    _json(output / "context_materialization_qa.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--calibration-axes", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--secondary-face", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--jpeg-temp-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--materialize-contexts-only", action="store_true")
    parser.add_argument("--wav2lip-batch-size", type=int, default=64)
    parser.add_argument("--syncnet-batch-size", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if _sha(Path(args.spec).resolve(strict=True)) != EXPECTED_RANDOM_SPEC_SHA256:
        raise ValueError("frozen random-specificity specification hash mismatch")
    if args.preflight_only and args.materialize_contexts_only:
        raise ValueError("choose only one preflight mode")
    if args.materialize_contexts_only:
        required_files = (
            args.spec, args.input_manifest, args.calibration_summary, args.calibration_axes,
            args.primary_face, args.secondary_face, args.wav2lip_checkpoint,
            args.syncnet_checkpoint, args.ffmpeg,
        )
        missing = [str(Path(value)) for value in required_files if not Path(value).is_file()]
        required_dirs = (args.source_run_dir, args.input_root, args.wav2lip_root, args.syncnet_root)
        missing.extend(str(Path(value)) for value in required_dirs if not Path(value).is_dir())
        if missing:
            raise FileNotFoundError(f"context materialization paths missing: {missing}")
        report = _materialize_contexts_only(args)
        print(json.dumps({key: report[key] for key in ("status", "context_count", "planned_row_count", "verified_context_count", "verified_array_count", "integrity_failure_count", "source_artifact_inventory_sha256")}, separators=(",", ":")))
        return 0 if report["status"] == "materialization_pass" else 3
    if args.preflight_only:
        required_files = (
            args.spec, args.input_manifest, args.calibration_summary, args.calibration_axes,
            args.primary_face, args.secondary_face, args.wav2lip_checkpoint,
            args.syncnet_checkpoint, args.ffmpeg,
        )
        missing = [str(Path(value)) for value in required_files if not Path(value).is_file()]
        required_dirs = (args.source_run_dir, args.input_root, args.wav2lip_root, args.syncnet_root)
        missing.extend(str(Path(value)) for value in required_dirs if not Path(value).is_dir())
        if missing:
            raise FileNotFoundError(f"preflight paths missing: {missing}")
        rows, artifacts, source = _read_source_inventory(args, write_contexts=False)
        output = Path(args.run_dir).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"preflight target must be empty: {output}")
        dose_preflight = {
            f"random{i}": {
                key: sum(row["random_dose_qc"][f"random{i}"][key] for row in artifacts)
                for key in (
                    "dose_contract_pass_frames", "dose_tolerance_miss_frames",
                    "noncapped_tolerance_miss_frames", "capped_infeasible_frames",
                )
            } | {
                "max_clip_abs_actual_float32_dose_error": max(
                    row["random_dose_qc"][f"random{i}"]["max_abs_actual_float32_dose_error"] for row in artifacts
                ),
                "max_clip_p95_abs_actual_float32_dose_error": max(
                    row["random_dose_qc"][f"random{i}"]["p95_abs_actual_float32_dose_error"] for row in artifacts
                ),
            }
            for i in (1, 2, 3)
        }
        noncapped_misses = sum(values["noncapped_tolerance_miss_frames"] for values in dose_preflight.values())
        status = "preflight_pass" if noncapped_misses == 0 else "preflight_failed_noncapped_dose_miss"
        print(json.dumps({"status": status, "classification": "non-effect source/input/dose preflight", "planned_samples": len(artifacts), "planned_rows": len(rows), "source_run_dir": source["source_run_dir"], "source_protocol_sha256": source["source_protocol_sha256"], "source_rows_sha256": source["source_rows_sha256"], "source_results_sha256": source["source_results_sha256"], "source_lineage": source["source_lineage"], "context_count": len(source["source_artifacts"]), "random_dose_preflight": dose_preflight}, separators=(",", ":")))
        return 0 if noncapped_misses == 0 else 2
    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
