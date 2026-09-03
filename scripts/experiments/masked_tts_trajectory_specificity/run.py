"""Trajectory-specificity diagnosis for the masked TTS reconstruction experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from scripts.experiments.masked_tts_reconstruction.config import (
    BATCH_SIZE,
    CPU_THREADS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MIN_TTS_FRAMES,
    SEEDS,
    TRAIN_STEPS,
    VELOCITY_WEIGHT,
    WAVLM_DIM,
    WEIGHT_DECAY,
)
from scripts.experiments.masked_tts_reconstruction.evaluate import array_hash, score_condition
from scripts.experiments.masked_tts_reconstruction.features import build_example, phone_phase_linear
from scripts.experiments.masked_tts_reconstruction.protocol import canonical_json, normalize_phone, sha256_file, sha256_text, write_json as write_protocol_json
from scripts.experiments.masked_tts_reconstruction.run import _build_examples, _load_features
from scripts.experiments.masked_tts_reconstruction.model import MaskedNaturalReconstructor, reconstruction_loss
from scripts.experiments.masked_tts_reconstruction.train import (
    clone_state,
    load_checkpoint,
    make_schedule,
    schedule_hash,
    set_deterministic_cpu,
    state_hash,
)
from scripts.experiments.masked_tts_tfg_probe.mel_drivers import MEL_MAX, MEL_MIN, patch_normalized_mel
from scripts.experiments.masked_tts_tfg_probe.run import (
    EXPECTED_SYNCNET_SHA256,
    EXPECTED_WAV2LIP_SHA256,
    MIN_TRACK,
    SYNCNET,
    SYNCNET_MODEL,
    SYNCNET_PY,
    WAV2LIP,
    WAV2LIP_CHECKPOINT,
    WAV2LIP_PY,
    _last_json,
    _parse_syncnet,
    _run_logged,
    render_all,
)
from scripts.experiments.masked_tts_tfg_probe.phase_a import read_json, write_json

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PARENT = REPO / "runs/lrs3_masked_tts_retention_exploratory_20260901"
DEFAULT_CONFIRMATION = REPO / "runs/lrs3_masked_tts_tfg_confirmation_20260902"
DEFAULT_RUN = REPO / "runs/lrs3_masked_tts_trajectory_specificity_20260902"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260902
EXPECTED_RECORDS = 16
EXPECTED_GROUPS = 8
CONTROL_CONDITIONS = ("SAME_PHONE_WRONG_INSTANCE", "WITHIN_PHONE_REVERSED")
BASE_MODEL_ARM = "paired_tts"
CLAIM_BOUNDARY = [
    "exploratory mechanism and engineering diagnostic on previously inspected records",
    "direct-mel frozen-Wav2Lip validation with untouched natural audio after strict replacement",
    "not audible TTS-feature retention",
    "not waveform reachability",
    "not audio quality",
    "not natural-prosody preservation",
    "not population generalization",
    "not a deployable replacement system",
]


def _write(path: Path, payload: Any, *, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        return
    write_json(path, payload)


def _array_hash(value: np.ndarray) -> str:
    return array_hash(np.asarray(value, dtype=np.float32))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {str(key): np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    if any(value.dtype.hasobject or not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"invalid NPZ arrays: {path}")
    return result


def _load_parent(parent: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    lock = read_json(parent / "00_lock/lock.json")
    masks = read_json(parent / "01_masks/mask_manifest.json")
    natural, tts, stats = _load_features(parent, lock, masks)
    if masks.get("status") not in (None, "complete") and masks.get("readiness", {}).get("sufficient") is not True:
        raise ValueError("parent mask manifest is not complete")
    return lock, masks, stats, natural, tts


def _cohort_records(cohort: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = list(cohort.get("records", []))
    if cohort.get("status") != "complete" or len(records) != EXPECTED_RECORDS:
        raise ValueError("confirmation cohort is incomplete")
    if len({str(row["sample_id"]) for row in records}) != len(records):
        raise ValueError("confirmation cohort has duplicate records")
    if len(cohort.get("groups", [])) != EXPECTED_GROUPS:
        raise ValueError("confirmation cohort has invalid group count")
    for row in records:
        hashes = [str(value) for value in row.get("evaluation_mask_sha256", [])]
        if len(hashes) != int(row.get("evaluation_mask_count", -1)) or not hashes:
            raise ValueError(f"confirmation cohort masks are incomplete for {row.get('sample_id')}")
    return records


def _mask_map(masks: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {str(row["mask_sha256"]): row for row in masks.get("masks", []) if row.get("prototype_split") == "evaluation"}
    if len(result) != sum(int(row.get("prototype_split") == "evaluation") for row in masks.get("masks", [])):
        raise ValueError("duplicate evaluation mask hash")
    return result


def _bootstrap(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (EXPECTED_GROUPS,) or not np.isfinite(array).all():
        raise ValueError("bootstrap requires eight finite source-group values")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    estimates = np.median(array[indices], axis=1)
    return {
        "median": float(np.median(array)),
        "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))],
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "unit": "whole_source_group",
        "method": "numpy_quantile_linear",
    }


def _metric_summary(group_rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in group_rows]
    result = _bootstrap(values)
    result["positive_groups"] = int(sum(value > 0 for value in values))
    result["values"] = values
    result["pass"] = bool(result["ci95"][0] > 0 and result["positive_groups"] >= 7)
    return result


def _aggregate_record_group(
    records: Sequence[Mapping[str, Any]],
    per_seed: Mapping[tuple[str, int], Mapping[str, float]],
    metric_names: Sequence[str],
    natural: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        seed_rows = [per_seed[(sid, int(seed))] for seed in SEEDS]
        row: dict[str, Any] = {"sample_id": sid, "source_group": str(record["source_group"]), "seed_count": len(seed_rows)}
        for metric in metric_names:
            row[metric] = float(np.median([float(seed_row[metric]) for seed_row in seed_rows]))
        if natural is not None:
            row.update(natural[sid])
        record_rows.append(row)
    groups = []
    seen = []
    for record in records:
        group = str(record["source_group"])
        if group in seen:
            continue
        seen.append(group)
        local = [row for row in record_rows if row["source_group"] == group]
        if len(local) != 2:
            raise ValueError(f"expected two records in source group {group}")
        groups.append({"source_group": group, "record_count": len(local), **{metric: float(np.median([float(row[metric]) for row in local])) for metric in metric_names}})
    return record_rows, groups


def _reference_audit(run_dir: Path, confirmation: Path, cohort: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    output = run_dir / "01_reference_audit"
    analysis_path = output / "analysis.json"
    if analysis_path.exists() and resume:
        return read_json(analysis_path)
    scores = read_json(confirmation / "05_syncnet/summary.json")
    records = _cohort_records(cohort)
    if scores.get("status") != "complete" or int(scores.get("score_count", -1)) != EXPECTED_RECORDS * 10:
        raise ValueError("confirmation score matrix is incomplete")
    by_cell: dict[tuple[str, int | None, str], Mapping[str, Any]] = {}
    for row in scores["scores"]:
        key = (str(row["sample_id"]), None if row.get("seed") is None else int(row["seed"]), str(row["condition"]))
        if key in by_cell:
            raise ValueError(f"duplicate confirmation cell: {key}")
        by_cell[key] = row
    metric_names = (
        "paired_reference_C",
        "paired_reference_D",
        "centroid_reference_C",
        "centroid_reference_D",
        "centroid_modality_C",
        "centroid_modality_D",
        "paired_modality_C",
        "paired_modality_D",
    )
    per_seed: dict[tuple[str, int], dict[str, float]] = {}
    natural_by_record: dict[str, dict[str, float]] = {}
    seed_rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        nat = by_cell.get((sid, None, "NATURAL_MEL"))
        if nat is None:
            raise ValueError(f"missing NATURAL_MEL score for {sid}")
        natural_by_record[sid] = {"natural_sync_c": float(nat["sync_c"]), "natural_sync_d": float(nat["sync_d"])}
        for seed in SEEDS:
            paired = by_cell.get((sid, int(seed), "PAIRED_TTS"))
            centroid = by_cell.get((sid, int(seed), "PHONE_CENTROID"))
            zero = by_cell.get((sid, int(seed), "NAT_ONLY"))
            if paired is None or centroid is None or zero is None:
                raise ValueError(f"missing confirmation condition for {sid} seed {seed}")
            values = {
                "paired_reference_C": float(paired["sync_c"]) - float(nat["sync_c"]),
                "paired_reference_D": float(nat["sync_d"]) - float(paired["sync_d"]),
                "centroid_reference_C": float(centroid["sync_c"]) - float(nat["sync_c"]),
                "centroid_reference_D": float(nat["sync_d"]) - float(centroid["sync_d"]),
                "centroid_modality_C": float(centroid["sync_c"]) - float(zero["sync_c"]),
                "centroid_modality_D": float(zero["sync_d"]) - float(centroid["sync_d"]),
                "paired_modality_C": float(paired["sync_c"]) - float(zero["sync_c"]),
                "paired_modality_D": float(zero["sync_d"]) - float(paired["sync_d"]),
            }
            per_seed[(sid, int(seed))] = values
            seed_rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "seed": int(seed), **values})
    record_rows, group_rows = _aggregate_record_group(records, per_seed, metric_names, natural_by_record)
    summary = {metric: _metric_summary(group_rows, metric) for metric in metric_names}
    paired_reference_pass = all(summary[metric]["pass"] for metric in ("paired_reference_C", "paired_reference_D"))
    centroid_practical_pass = all(summary[metric]["pass"] for metric in ("centroid_modality_C", "centroid_modality_D"))
    status = "PAIRED_BEATS_NATURAL_REFERENCE" if paired_reference_pass else "PAIRED_NOT_SHOWN_TO_BEAT_NATURAL_REFERENCE"
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "part": "A",
        "engineering_audit": status,
        "retrospective": True,
        "previously_inspected_records": True,
        "record_count": EXPECTED_RECORDS,
        "group_count": EXPECTED_GROUPS,
        "score_count": int(scores["score_count"]),
        "cohort_manifest_sha256": sha256_file(confirmation / "00_cohort/manifest.json"),
        "score_manifest_sha256": sha256_file(confirmation / "05_syncnet/summary.json"),
        "seed_rows": seed_rows,
        "record_rows": record_rows,
        "group_rows": group_rows,
        "bootstrap": summary,
        "rules": {
            "reference": "both lower bounds > 0 and at least 7/8 positive source groups on Sync-C and Sync-D",
            "centroid_practical": "PHONE_CENTROID over NAT_ONLY uses the same rule",
            "aggregation": "seed median, record median, two-record source-group median",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "sealed_splits_accessed": False,
    }
    _write(analysis_path, analysis, resume=resume)
    report = [
        "# Part A: natural-mel reference audit",
        "",
        f"Status: **{status}**.",
        "",
        "This is a retrospective engineering audit of the previously inspected 16-record confirmation matrix. No cells were rerendered or rescored, and the prior confirmation decision is unchanged.",
        "",
        "| Contrast | Median | 95% CI | Positive groups | Pass |",
        "|---|---:|---|---:|---|",
    ]
    labels = {
        "paired_reference_C": "PAIRED_TTS - NATURAL_MEL (Sync-C)",
        "paired_reference_D": "NATURAL_MEL - PAIRED_TTS (Sync-D)",
        "centroid_reference_C": "PHONE_CENTROID - NATURAL_MEL (Sync-C)",
        "centroid_reference_D": "NATURAL_MEL - PHONE_CENTROID (Sync-D)",
        "centroid_modality_C": "PHONE_CENTROID - NAT_ONLY (Sync-C)",
        "centroid_modality_D": "NAT_ONLY - PHONE_CENTROID (Sync-D)",
        "paired_modality_C": "PAIRED_TTS - NAT_ONLY (Sync-C)",
        "paired_modality_D": "NAT_ONLY - PAIRED_TTS (Sync-D)",
    }
    for metric in metric_names:
        row = summary[metric]
        report.append(f"| {labels[metric]} | {row['median']:.5f} | [{row['ci95'][0]:.5f}, {row['ci95'][1]:.5f}] | {row['positive_groups']}/8 | {'yes' if row['pass'] else 'no'} |")
    report.extend(["", "The non-pass reference label does not prove that natural mel is superior. The earlier `CONFIRMED_MODALITY_ONLY` decision remains unchanged.", "", "## Claim boundary", "", *[f"- {item}" for item in CLAIM_BOUNDARY], ""])
    _write(output / "report.md", "\n".join(report), resume=resume)
    return analysis


def _eligible_donors(target: Mapping[str, Any], train_masks: Iterable[Mapping[str, Any]], tts: Mapping[str, np.ndarray]) -> list[Mapping[str, Any]]:
    target_label = normalize_phone(target["label"])
    target_sid = str(target["sample_id"])
    target_group = str(target["source_group"])
    candidates = []
    for row in train_masks:
        sid = str(row["sample_id"])
        start = int(row["tts_frame_start"])
        end = int(row["tts_frame_end"])
        values = tts.get(sid)
        if normalize_phone(row["label"]) != target_label or sid == target_sid or str(row["source_group"]) == target_group:
            continue
        if values is None or values.ndim != 2 or values.shape[1] != WAVLM_DIM or start < 0 or end > values.shape[0] or end - start < MIN_TTS_FRAMES:
            continue
        candidates.append(row)
    return sorted(candidates, key=lambda row: (str(row["source_group"]), str(row["sample_id"]), int(row["canonical_index"]), str(row["mask_sha256"])))


def _select_donor(target: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], int]:
    if not candidates:
        raise ValueError("no eligible same-phone donor")
    index = int(str(target["mask_sha256"])[:16], 16) % len(candidates)
    return candidates[index], index


def _reverse_inside_core(features: np.ndarray, core_start: int, core_end: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.shape != (96, WAVLM_DIM) or core_start < 0 or core_end > values.shape[0] or core_start >= core_end:
        raise ValueError("invalid trajectory core")
    result = values.copy()
    result[core_start:core_end] = result[core_start:core_end][::-1].copy()
    result = np.ascontiguousarray(result, dtype=np.float32)
    outside = np.ones(result.shape[0], dtype=bool)
    outside[core_start:core_end] = False
    if np.any(result[outside] != 0.0):
        raise ValueError("reversed trajectory has nonzero outside-core values")
    return result


def build_controls(
    run_dir: Path,
    parent: Path,
    confirmation: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = run_dir / "02_controls/manifest.json"
    features_path = run_dir / "02_controls/features.npz"
    if manifest_path.exists() and features_path.exists() and resume:
        return read_json(manifest_path)
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    records = _cohort_records(cohort)
    lock, masks, stats, natural, tts = _load_parent(parent)
    del lock
    mask_by_hash = _mask_map(masks)
    train_masks = [row for row in masks["masks"] if row.get("prototype_split") == "train"]
    control_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for record in records:
        for mask_hash in record["evaluation_mask_sha256"]:
            target = mask_by_hash.get(str(mask_hash))
            if target is None:
                invalid.append({"mask_sha256": str(mask_hash), "reason": "evaluation mask missing from parent manifest"})
                continue
            sid = str(target["sample_id"])
            paired = build_example(target, natural, tts, stats)
            paired_features = np.asarray(paired["tts_features"], dtype=np.float32)
            donors = _eligible_donors(target, train_masks, tts)
            if not donors:
                invalid.append({"mask_sha256": str(mask_hash), "reason": "no eligible same-phone donor"})
                continue
            donor, index = _select_donor(target, donors)
            wrong = build_example(target, natural, tts, stats, tts_mask=donor)
            wrong_features = np.asarray(wrong["tts_features"], dtype=np.float32)
            core_start, core_end = int(target["core_start"]), int(target["core_end"])
            reversed_features = _reverse_inside_core(paired_features, core_start, core_end)
            outside = np.ones(reversed_features.shape[0], dtype=bool)
            outside[core_start:core_end] = False
            if np.any(reversed_features[outside] != 0.0):
                invalid.append({"mask_sha256": str(mask_hash), "reason": "reversed trajectory has nonzero outside-core values"})
                continue
            paired_hash = _array_hash(paired_features)
            wrong_hash = _array_hash(wrong_features)
            reversed_hash = _array_hash(reversed_features)
            if wrong_hash == paired_hash or reversed_hash == paired_hash:
                invalid.append({"mask_sha256": str(mask_hash), "reason": "control feature hash equals paired hash"})
                continue
            wrong_key = f"{mask_hash}__SAME_PHONE_WRONG_INSTANCE"
            reversed_key = f"{mask_hash}__WITHIN_PHONE_REVERSED"
            control_arrays[wrong_key] = wrong_features
            control_arrays[reversed_key] = reversed_features
            rows.append({
                "mask_sha256": str(mask_hash),
                "sample_id": sid,
                "source_group": str(target["source_group"]),
                "label": str(target["label"]),
                "normalized_label": normalize_phone(target["label"]),
                "core_start": core_start,
                "core_end": core_end,
                "destination_length": core_end - core_start,
                "paired_feature_sha256": paired_hash,
                "controls": {
                    "SAME_PHONE_WRONG_INSTANCE": {
                        "feature_key": wrong_key,
                        "feature_sha256": wrong_hash,
                        "source_sample_id": str(donor["sample_id"]),
                        "source_group": str(donor["source_group"]),
                        "source_mask_sha256": str(donor["mask_sha256"]),
                        "selection_index": index,
                        "candidate_count": len(donors),
                        "selection_rule": "first_16_hex_mask_hash_modulo_sorted_identity_only_candidates",
                        "build_path": "build_example_tts_mask_then_phone_phase_linear",
                    },
                    "WITHIN_PHONE_REVERSED": {
                        "feature_key": reversed_key,
                        "feature_sha256": reversed_hash,
                        "source_sample_id": sid,
                        "source_group": str(target["source_group"]),
                        "source_mask_sha256": str(mask_hash),
                        "transform": "reverse_paired_aligned_frames_inside_target_core_only",
                    },
                },
            })
    expected = sum(len(row["evaluation_mask_sha256"]) for row in records)
    complete = not invalid and len(rows) == expected and len(control_arrays) == expected * 2
    if not complete:
        manifest = {
            "schema_version": 1,
            "manifest_type": "lrs3_masked_tts_trajectory_controls",
            "status": "not_evaluated",
            "reason": "control construction or provenance incomplete",
            "expected_masks": expected,
            "evaluated_masks": len(rows),
            "invalid": invalid,
            "records": rows,
            "sealed_splits_accessed": False,
        }
        _write(manifest_path, manifest, resume=resume)
        raise ValueError("Part B control construction is incomplete; see control manifest")
    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(features_path, **control_arrays)
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_trajectory_controls",
        "status": "complete",
        "parent_mask_manifest_sha256": sha256_file(parent / "01_masks/mask_manifest.json"),
        "confirmation_cohort_sha256": sha256_file(confirmation / "00_cohort/manifest.json"),
        "feature_file_sha256": sha256_file(features_path),
        "record_count": EXPECTED_RECORDS,
        "mask_count": expected,
        "feature_count": len(control_arrays),
        "conditions": list(CONTROL_CONDITIONS),
        "records": rows,
        "invalid": [],
        "donor_pool": "frozen parent prototype_split=train masks only",
        "selection_does_not_use": ["loss", "prediction", "Wav2Lip", "SyncNet"],
        "sealed_splits_accessed": False,
    }
    _write(manifest_path, manifest, resume=resume)
    return manifest


def _control_examples(
    mask: Mapping[str, Any],
    feature_map: Mapping[str, np.ndarray],
    control_row: Mapping[str, Any],
    natural: Mapping[str, np.ndarray],
    tts: Mapping[str, np.ndarray],
    stats: Mapping[str, Any],
    condition: str,
) -> dict[str, np.ndarray]:
    base = build_example(mask, natural, tts, stats)
    key = str(control_row["controls"][condition]["feature_key"])
    example = {key_name: np.asarray(base[key_name], dtype=np.float32) for key_name in ("natural_mel", "masked_support", "target_core", "target")}
    example["tts_features"] = np.asarray(feature_map[key], dtype=np.float32)
    return example


def run_reconstruction(
    run_dir: Path,
    parent: Path,
    confirmation: Path,
    controls: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output = run_dir / "03_frozen_diagnosis"
    result_path = output / "reconstruction.json"
    if result_path.exists() and resume:
        return read_json(result_path)
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    records = _cohort_records(cohort)
    lock, masks, stats, natural, tts = _load_parent(parent)
    mask_by_hash = _mask_map(masks)
    feature_map = _load_npz(run_dir / "02_controls/features.npz")
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        model, checkpoint_payload = load_checkpoint(parent / "03_train" / str(seed) / BASE_MODEL_ARM / "checkpoint.pt")
        for record in records:
            for mask_hash in record["evaluation_mask_sha256"]:
                target = mask_by_hash[str(mask_hash)]
                parent_record_path = parent / "04_eval" / str(seed) / str(mask_hash) / "PAIRED_TTS" / "record.json"
                parent_record = read_json(parent_record_path)
                paired_loss = dict(parent_record["loss"])
                control_row = next(row for row in controls["records"] if row["mask_sha256"] == str(mask_hash))
                for condition in CONTROL_CONDITIONS:
                    example = _control_examples(target, feature_map, control_row, natural, tts, stats, condition)
                    prediction, loss = score_condition(model, example, zero_tts=False)
                    cell = output / "reconstruction" / str(seed) / str(mask_hash) / condition
                    cell.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cell / "prediction.npz", prediction=prediction)
                    record_payload = {
                        "schema_version": 1,
                        "seed": int(seed),
                        "sample_id": str(target["sample_id"]),
                        "source_group": str(target["source_group"]),
                        "mask_sha256": str(mask_hash),
                        "condition": condition,
                        "status": "complete",
                        "loss": loss,
                        "paired_loss": paired_loss,
                        "reconstruction_gain": float(loss["total"]) - float(paired_loss["total"]),
                        "prediction_shape": list(prediction.shape),
                        "prediction_sha256": _array_hash(prediction),
                        "parent_paired_prediction_sha256": str(parent_record["prediction_sha256"]),
                        "control_feature_sha256": str(control_row["controls"][condition]["feature_sha256"]),
                        "checkpoint_sha256": sha256_file(parent / "03_train" / str(seed) / BASE_MODEL_ARM / "checkpoint.pt"),
                        "checkpoint_payload_initial_state_sha256": checkpoint_payload["initial_state_sha256"],
                    }
                    write_json(cell / "record.json", record_payload)
                    rows.append(record_payload)
        print(f"reconstruction seed {seed} complete", flush=True)
    expected = sum(len(row["evaluation_mask_sha256"]) for row in records) * len(SEEDS) * len(CONTROL_CONDITIONS)
    if len(rows) != expected:
        raise ValueError(f"reconstruction matrix incomplete: {len(rows)}/{expected}")
    result = {
        "schema_version": 1,
        "status": "complete",
        "record_count": EXPECTED_RECORDS,
        "mask_count": expected // (len(SEEDS) * len(CONTROL_CONDITIONS)),
        "seed_count": len(SEEDS),
        "condition_count": len(CONTROL_CONDITIONS),
        "required_cells": expected,
        "conditions": list(CONTROL_CONDITIONS),
        "records": rows,
        "parent_evaluation_sha256": sha256_file(parent / "04_eval/evaluation.json"),
        "sealed_splits_accessed": False,
    }
    _write(result_path, result, resume=resume)
    return result


def _load_natural_mels(path: Path) -> dict[str, np.ndarray]:
    return _load_npz(path)


def build_control_drivers(
    run_dir: Path,
    parent: Path,
    confirmation: Path,
    controls: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output_dir = run_dir / "03_frozen_diagnosis/mel_drivers"
    manifest_path = output_dir / "drivers.json"
    if manifest_path.exists() and resume:
        return read_json(manifest_path)
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    records = _cohort_records(cohort)
    _, masks, stats, natural_mels, _ = _load_parent(parent)
    mask_by_hash = _mask_map(masks)
    recon_by_key = {(int(row["seed"]), str(row["mask_sha256"]), str(row["condition"])): row for row in reconstruction["records"]}
    driver_rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        natural = np.asarray(natural_mels[sid], dtype=np.float32)
        mean = np.asarray(stats["mel_mean"], dtype=np.float32)[:, None]
        std = np.asarray(stats["mel_std"], dtype=np.float32)[:, None]
        natural_standardized = (natural - mean) / std
        for seed in SEEDS:
            contributions_by_condition = {condition: [] for condition in CONTROL_CONDITIONS}
            used_by_condition = {condition: [] for condition in CONTROL_CONDITIONS}
            for mask_hash in record["evaluation_mask_sha256"]:
                mask = mask_by_hash[str(mask_hash)]
                global_start = int(mask["window_start_frame"]) + int(mask["core_start"])
                global_end = int(mask["window_start_frame"]) + int(mask["core_end"])
                if global_start < 0 or global_end > natural.shape[1] or global_start >= global_end:
                    raise ValueError(f"control prediction core outside natural mel for {mask_hash}")
                for condition in CONTROL_CONDITIONS:
                    recon = recon_by_key[(int(seed), str(mask_hash), condition)]
                    prediction_path = run_dir / "03_frozen_diagnosis/reconstruction" / str(seed) / str(mask_hash) / condition / "prediction.npz"
                    with np.load(prediction_path, allow_pickle=False) as archive:
                        prediction = np.asarray(archive["prediction"], dtype=np.float32)
                    if _array_hash(prediction) != str(recon["prediction_sha256"]):
                        raise ValueError(f"control prediction hash changed: {prediction_path}")
                    core_start, core_end = int(mask["core_start"]), int(mask["core_end"])
                    contributions_by_condition[condition].append((global_start, global_end, prediction[core_start:core_end]))
                    used_by_condition[condition].append({
                        "mask_sha256": str(mask_hash),
                        "global_start_frame": global_start,
                        "global_end_frame": global_end,
                        "core_start": core_start,
                        "core_end": core_end,
                        "prediction_sha256": str(recon["prediction_sha256"]),
                    })
            for condition in CONTROL_CONDITIONS:
                candidate, counts = patch_normalized_mel(natural_standardized, contributions_by_condition[condition])
                denormalized = candidate * std + mean
                output = natural.copy()
                covered = counts > 0
                output[:, covered] = denormalized[:, covered]
                unclamped = output.copy()
                clamp_mask = (output < MEL_MIN) | (output > MEL_MAX)
                output = np.clip(output, MEL_MIN, MEL_MAX).astype(np.float32, copy=False)
                if np.any(output[:, ~covered] != natural[:, ~covered]):
                    raise ValueError(f"control driver changed outside-core frames for {sid} {condition}")
                driver_id = f"{sid}__{condition}__{seed}"
                path = output_dir / f"{driver_id}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, output, allow_pickle=False)
                driver_rows.append({
                    "driver_id": driver_id,
                    "sample_id": sid,
                    "source_group": str(record["source_group"]),
                    "condition": condition,
                    "seed": int(seed),
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "shape": list(output.shape),
                    "natural_mel_sha256": _array_hash(natural),
                    "clamp_fraction": float(np.mean(clamp_mask)),
                    "clamped_values": int(np.count_nonzero(clamp_mask)),
                    "total_values": int(output.size),
                    "mel_range_before_clamp": [float(unclamped.min()), float(unclamped.max())],
                    "mel_range_after_clamp": [float(output.min()), float(output.max())],
                    "covered_frame_count": int(np.count_nonzero(covered)),
                    "overlap_frame_count": int(np.count_nonzero(counts > 1)),
                    "used_masks": used_by_condition[condition],
                    "outside_core_policy": "exact_natural_mel_source",
                })
    expected = EXPECTED_RECORDS * len(SEEDS) * len(CONTROL_CONDITIONS)
    if len(driver_rows) != expected:
        raise ValueError(f"control driver matrix incomplete: {len(driver_rows)}/{expected}")
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_trajectory_direct_mel_drivers",
        "status": "complete",
        "parent_run": str(parent),
        "cohort_manifest_sha256": sha256_file(confirmation / "00_cohort/manifest.json"),
        "reconstruction_sha256": sha256_file(run_dir / "03_frozen_diagnosis/reconstruction.json"),
        "natural_mel_source_sha256": sha256_file(parent / "02_features/natural_mels.npz"),
        "normalization_sha256": sha256_file(parent / "02_features/normalization.json"),
        "conditions": list(CONTROL_CONDITIONS),
        "seeds": list(SEEDS),
        "record_count": EXPECTED_RECORDS,
        "drivers_per_record": len(SEEDS) * len(CONTROL_CONDITIONS),
        "driver_count": len(driver_rows),
        "overlap_policy": "arithmetic_mean_of_contributing_prediction_cores",
        "outside_core_policy": "exact_natural_mel_source",
        "valid_mel_range": [MEL_MIN, MEL_MAX],
        "drivers": driver_rows,
        "sealed_splits_accessed": False,
    }
    _write(manifest_path, manifest, resume=resume)
    return manifest


def render_controls_parallel(
    run_dir: Path,
    cohort_path: Path,
    drivers_path: Path,
    parity_path: Path,
    parent_render_path: Path,
    render_root: Path,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("worker count must be positive")
    parity = read_json(parity_path)
    if parity.get("status") != "PASS":
        raise ValueError("natural-mel parity has not passed")
    if sha256_file(WAV2LIP_CHECKPOINT) != EXPECTED_WAV2LIP_SHA256:
        raise ValueError("Wav2Lip checkpoint hash changed")
    cohort = read_json(cohort_path)
    drivers = read_json(drivers_path)
    parent_renders = read_json(parent_render_path)
    records_by_id = {str(row["sample_id"]): row for row in cohort["records"]}
    boxes_by_sample = {str(row["sample_id"]): str(row["boxes"]) for row in parent_renders.get("renders", [])}
    expected_count = int(drivers.get("driver_count", -1))
    driver_rows = list(drivers.get("drivers", []))
    if expected_count <= 0 or len(driver_rows) != expected_count:
        raise ValueError("control driver manifest is incomplete")
    if any(not Path(path).is_file() for path in boxes_by_sample.values()):
        raise ValueError("parent face boxes are incomplete")
    render_root = render_root.resolve()

    def render_one(driver: Mapping[str, Any]) -> dict[str, Any]:
        sample_id = str(driver["sample_id"])
        record = records_by_id[sample_id]
        output = render_root / "videos" / f"{driver['driver_id']}.mp4"
        boxes = Path(boxes_by_sample[sample_id])
        log = render_root / "logs" / f"{driver['driver_id']}.log"
        command = [
            str(WAV2LIP_PY), str(REPO / "scripts/experiments/masked_tts_tfg_probe/direct_mel.py"), "render",
            "--checkpoint", str(WAV2LIP_CHECKPOINT), "--face", str(record["face"]), "--mel", str(driver["path"]),
            "--outfile", str(output), "--face-det-batch-size", "4", "--wav2lip-batch-size", "4", "--nosmooth",
            "--boxes-input", str(boxes),
        ]
        if not output.is_file() or output.stat().st_size == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            text = _run_logged(command, REPO, log)
            direct_result = _last_json(text)
        else:
            direct_result = {"outfile": str(output), "resumed": True}
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"missing direct-mel render: {output}")
        return {
            "driver_id": driver["driver_id"],
            "sample_id": sample_id,
            "source_group": record["source_group"],
            "condition": driver["condition"],
            "seed": driver["seed"],
            "driver_sha256": driver["sha256"],
            "face": record["face"],
            "face_sha256": record["face_sha256"],
            "video": str(output.resolve()),
            "video_sha256": sha256_file(output),
            "boxes": str(boxes.resolve()),
            "log": str(log.resolve()),
            "command": command,
            "direct_result": direct_result,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(render_one, driver_rows))
    for index, row in enumerate(rows, 1):
        print(f"rendered {index}/{expected_count} {row['driver_id']}", flush=True)
    result = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_direct_mel_wav2lip_renders",
        "status": "complete",
        "cohort_manifest_sha256": sha256_file(cohort_path),
        "drivers_manifest_sha256": sha256_file(drivers_path),
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
        "render_count": len(rows),
        "renders": rows,
        "worker_count": workers,
        "boxes_source_render_manifest_sha256": sha256_file(parent_render_path),
        "sealed_splits_accessed": False,
    }
    write_protocol_json(render_root / "render_manifest.json", result, overwrite=True)
    return result


def score_controls_parallel(
    run_dir: Path,
    cohort_path: Path,
    render_path: Path,
    *,
    workers: int = 6,
) -> dict[str, Any]:
    from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import mux_and_verify

    if workers < 1:
        raise ValueError("worker count must be positive")
    if sha256_file(SYNCNET_MODEL) != EXPECTED_SYNCNET_SHA256:
        raise ValueError("SyncNet model hash changed")
    cohort = read_json(cohort_path)
    renders = read_json(render_path)
    records_by_id = {str(row["sample_id"]): row for row in cohort["records"]}
    expected_count = int(renders.get("render_count", -1))
    render_rows = list(renders.get("renders", []))
    if expected_count <= 0 or len(render_rows) != expected_count:
        raise ValueError("render manifest is incomplete")
    replacement_dir = run_dir / "04_replacement"
    sync_dir = run_dir / "05_syncnet"
    replacement_dir.mkdir(parents=True, exist_ok=True)
    sync_dir.mkdir(parents=True, exist_ok=True)

    def score_one(render: Mapping[str, Any]) -> dict[str, Any]:
        sample_id = str(render["sample_id"])
        record = records_by_id[sample_id]
        muxed = replacement_dir / f"{render['driver_id']}.mkv"
        if muxed.is_file():
            mux = {"path": str(muxed.resolve()), "sha256": sha256_file(muxed), "resumed": True}
        else:
            mux = mux_and_verify(
                source_video=Path(str(render["video"])),
                expected_audio=Path(str(record["natural_audio"])),
                output_path=muxed,
            )
        cell_sync_dir = sync_dir / str(render["driver_id"])
        cell_sync_dir.mkdir(parents=True, exist_ok=True)
        reference = f"masked_tts_tfg_{sample_id}"
        pipeline_log = sync_dir / "logs" / f"{render['driver_id']}.pipeline.log"
        score_log = sync_dir / "logs" / f"{render['driver_id']}.score.log"
        parsed: dict[str, float | int] | None = None
        if score_log.is_file():
            try:
                parsed = _parse_syncnet(score_log)
            except ValueError:
                parsed = None
        if parsed is None:
            if not (cell_sync_dir / "syncnet_v2.model").exists():
                _run_logged([
                    str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(muxed), "--reference", reference,
                    "--data_dir", str(cell_sync_dir), "--min_track", str(MIN_TRACK), "--overwrite",
                ], SYNCNET, pipeline_log)
            _run_logged([
                str(SYNCNET_PY), "run_syncnet.py", "--videofile", str(muxed), "--reference", reference,
                "--data_dir", str(cell_sync_dir), "--initial_model", str(SYNCNET_MODEL),
            ], SYNCNET, score_log)
            parsed = _parse_syncnet(score_log)
        return {
            "driver_id": render["driver_id"],
            "sample_id": sample_id,
            "source_group": record["source_group"],
            "condition": render["condition"],
            "seed": render["seed"],
            "driver_sha256": render["driver_sha256"],
            "render_video_sha256": render["video_sha256"],
            "replacement": mux,
            "replacement_path": str(muxed.resolve()),
            "replacement_sha256": sha256_file(muxed),
            "natural_audio": record["natural_audio"],
            "natural_audio_sha256": record["natural_audio_sha256"],
            "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
            "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
            "min_track": MIN_TRACK,
            "reference": reference,
            "pipeline_log": str(pipeline_log.resolve()),
            "score_log": str(score_log.resolve()),
            **parsed,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(score_one, render_rows))
    if len(rows) != expected_count:
        raise ValueError(f"SyncNet matrix incomplete: {len(rows)}/{expected_count}")
    result = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_direct_mel_replacement_syncnet",
        "status": "complete",
        "cohort_manifest_sha256": sha256_file(cohort_path),
        "render_manifest_sha256": sha256_file(render_path),
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
        "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
        "min_track": MIN_TRACK,
        "score_count": len(rows),
        "scores": rows,
        "primary_audio": "untouched_natural_audio",
        "worker_count": workers,
        "sealed_splits_accessed": False,
    }
    write_json(sync_dir / "summary.json", result)
    return result


def _aggregate_contrast(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    paired_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    value_fn: Any,
    metric_name: str,
) -> dict[str, Any]:
    per_seed_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["sample_id"]), int(row["seed"]))
        per_seed_values[key].append(float(value_fn(row, paired_rows[key])))
    per_seed = {key: float(np.median(values)) for key, values in per_seed_values.items()}
    record_rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        values = [per_seed[(sid, int(seed))] for seed in SEEDS]
        record_rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "seed_values": values, metric_name: float(np.median(values))})
    group_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        group = str(record["source_group"])
        if group in seen:
            continue
        seen.add(group)
        local = [row for row in record_rows if row["source_group"] == group]
        group_rows.append({"source_group": group, "record_count": len(local), metric_name: float(np.median([float(row[metric_name]) for row in local]))})
    summary = _metric_summary(group_rows, metric_name)
    return {"record_rows": record_rows, "group_rows": group_rows, "summary": summary}


def analyze_frozen(
    run_dir: Path,
    confirmation: Path,
    controls: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    syncnet: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output = run_dir / "03_frozen_diagnosis"
    analysis_path = output / "analysis.json"
    if analysis_path.exists() and resume:
        return read_json(analysis_path)
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    records = _cohort_records(cohort)
    recon_rows = list(reconstruction.get("records", []))
    if reconstruction.get("status") != "complete":
        raise ValueError("reconstruction matrix is incomplete")
    paired_recon: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in recon_rows:
        key = (str(row["sample_id"]), int(row["seed"]))
        paired_recon.setdefault(key, row)
    if len(paired_recon) != EXPECTED_RECORDS * len(SEEDS):
        raise ValueError("paired reconstruction references are incomplete")
    recon_summary: dict[str, Any] = {}
    for condition in CONTROL_CONDITIONS:
        local = [row for row in recon_rows if str(row["condition"]) == condition]
        recon_summary[condition] = _aggregate_contrast(
            records, local, paired_recon,
            lambda row, paired: float(row["loss"]["total"]) - float(paired["paired_loss"]["total"]),
            "reconstruction_gain",
        )
    score_rows = list(syncnet.get("scores", []))
    if syncnet.get("status") != "complete" or len(score_rows) != EXPECTED_RECORDS * len(SEEDS) * len(CONTROL_CONDITIONS):
        raise ValueError("control SyncNet matrix is incomplete")
    confirmation_scores = read_json(confirmation / "05_syncnet/summary.json")["scores"]
    paired_sync: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in confirmation_scores:
        if str(row["condition"]) == "PAIRED_TTS":
            key = (str(row["sample_id"]), int(row["seed"]))
            paired_sync[key] = row
    if len(paired_sync) != EXPECTED_RECORDS * len(SEEDS):
        raise ValueError("paired SyncNet references are incomplete")
    sync_summary: dict[str, Any] = {}
    for condition in CONTROL_CONDITIONS:
        local = [row for row in score_rows if str(row["condition"]) == condition]
        sync_summary[condition] = {
            "Sync-C": _aggregate_contrast(records, local, paired_sync, lambda row, paired: float(paired["sync_c"]) - float(row["sync_c"]), "sync_c_gain"),
            "Sync-D": _aggregate_contrast(records, local, paired_sync, lambda row, paired: float(row["sync_d"]) - float(paired["sync_d"]), "sync_d_gain"),
        }
    reconstruction_pass = all(recon_summary[condition]["summary"]["pass"] for condition in CONTROL_CONDITIONS)
    sync_pass = all(sync_summary[condition][metric]["summary"]["pass"] for condition in CONTROL_CONDITIONS for metric in ("Sync-C", "Sync-D"))
    if reconstruction_pass and sync_pass:
        status = "TFG_TRAJECTORY_SIGNAL"
    elif reconstruction_pass:
        status = "RECONSTRUCTION_TRAJECTORY_SIGNAL_ONLY"
    else:
        status = "NO_TRAJECTORY_SIGNAL"
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "part": "B",
        "trajectory_status": status,
        "previously_inspected_records": True,
        "record_count": EXPECTED_RECORDS,
        "group_count": EXPECTED_GROUPS,
        "mask_count": int(reconstruction["mask_count"]),
        "reconstruction": recon_summary,
        "syncnet": sync_summary,
        "rules": {
            "contrast": "reconstruction control - paired loss; Sync-C paired - control; Sync-D control - paired",
            "pass": "95% CI lower bound > 0 and at least 7/8 positive source groups",
            "aggregation": "mask median within seed and record, seed median per record, two-record source-group median",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "sealed_splits_accessed": False,
    }
    _write(analysis_path, analysis, resume=resume)
    report = ["# Part B: frozen trajectory controls", "", f"Status: **{status}**.", "", "The exact paired trajectory was compared with a deterministic same-phone wrong-instance trajectory and a within-phone reversed trajectory. All downstream cells use untouched natural audio after strict replacement.", "", "## Reconstruction", "", "| Control | Median gain | 95% CI | Positive groups | Pass |", "|---|---:|---|---:|---|"]
    for condition in CONTROL_CONDITIONS:
        item = recon_summary[condition]["summary"]
        report.append(f"| {condition} | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if item['pass'] else 'no'} |")
    report.extend(["", "## SyncNet", "", "| Control | Metric | Median gain | 95% CI | Positive groups | Pass |", "|---|---|---:|---|---:|---|"])
    for condition in CONTROL_CONDITIONS:
        for metric in ("Sync-C", "Sync-D"):
            item = sync_summary[condition][metric]["summary"]
            report.append(f"| {condition} | {metric} | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if item['pass'] else 'no'} |")
    report.extend(["", "## Claim boundary", "", *[f"- {item}" for item in CLAIM_BOUNDARY], ""])
    _write(output / "report.md", "\n".join(report), resume=resume)
    return analysis


def run_binding(run_dir: Path, parent: Path, confirmation: Path, *, resume: bool) -> dict[str, Any]:
    path = run_dir / "00_binding/manifest.json"
    if path.exists() and resume:
        return read_json(path)
    cohort_path = confirmation / "00_cohort/manifest.json"
    files = {
        "parent_lock": parent / "00_lock/lock.json",
        "parent_masks": parent / "01_masks/mask_manifest.json",
        "parent_features": parent / "02_features/normalization.json",
        "parent_evaluation": parent / "04_eval/evaluation.json",
        "confirmation_cohort": cohort_path,
        "confirmation_scores": confirmation / "05_syncnet/summary.json",
        "confirmation_analysis": confirmation / "05_analysis/analysis.json",
    }
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError([str(path) for path in files.values() if not path.is_file()])
    cohort = read_json(cohort_path)
    records = _cohort_records(cohort)
    binding = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_trajectory_specificity_binding",
        "status": "complete",
        "run": "lrs3_masked_tts_trajectory_specificity_20260902",
        "parent_run": str(parent.resolve()),
        "confirmation_run": str(confirmation.resolve()),
        "files": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in files.items()},
        "record_count": len(records),
        "source_groups": [str(value) for value in cohort["groups"]],
        "sample_ids": [str(row["sample_id"]) for row in records],
        "seeds": list(SEEDS),
        "base_checkpoint_role": "full_correct",
        "effective_parent_checkpoint_arm": BASE_MODEL_ARM,
        "prior_conditions": ["NATURAL_MEL", "PAIRED_TTS", "PHONE_CENTROID", "NAT_ONLY"],
        "new_conditions": list(CONTROL_CONDITIONS),
        "previously_inspected_data_warning": "All 16 confirmation records were already inspected in the parent confirmation; this change is exploratory and mechanism-diagnostic, not an independent confirmation.",
        "selection_does_not_use": ["loss", "prediction quality", "Wav2Lip", "SyncNet"],
        "sealed_splits_accessed": False,
    }
    _write(path, binding, resume=resume)
    return binding


def _aggregate_absolute(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    value_fn: Any,
    metric_name: str,
) -> dict[str, Any]:
    by_seed_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_seed_values[(str(row["sample_id"]), int(row["seed"]))].append(float(value_fn(row)))
    by_seed = {key: float(np.median(values)) for key, values in by_seed_values.items()}
    record_rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        values = [by_seed[(sid, int(seed))] for seed in SEEDS]
        record_rows.append({"sample_id": sid, "source_group": str(record["source_group"]), metric_name: float(np.median(values))})
    group_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        group = str(record["source_group"])
        if group in seen:
            continue
        seen.add(group)
        local = [row for row in record_rows if row["source_group"] == group]
        group_rows.append({"source_group": group, "record_count": len(local), metric_name: float(np.median([float(row[metric_name]) for row in local]))})
    return {"record_rows": record_rows, "group_rows": group_rows, "summary": _metric_summary(group_rows, metric_name)}


def _batch_examples(examples: Sequence[Mapping[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    batch = {
        key: torch.from_numpy(np.stack([np.asarray(example[key], dtype=np.float32) for example in examples], axis=0))
        for key in ("natural_mel", "masked_support", "target_core", "target", "tts_features")
    }
    return batch


def _build_train_pairs(
    masks: Mapping[str, Any],
    natural: Mapping[str, np.ndarray],
    tts: Mapping[str, np.ndarray],
    stats: Mapping[str, Any],
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    train_masks = [row for row in masks["masks"] if row.get("prototype_split") == "train"]
    pairs: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for target in train_masks:
        mask_hash = str(target["mask_sha256"])
        donors = _eligible_donors(target, train_masks, tts)
        donor, _ = _select_donor(target, donors)
        paired_raw = build_example(target, natural, tts, stats)
        wrong_raw = build_example(target, natural, tts, stats, tts_mask=donor)
        paired = {key: np.asarray(paired_raw[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "target", "tts_features")}
        wrong = {key: np.asarray(wrong_raw[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "target", "tts_features")}
        pairs[mask_hash] = (paired, wrong)
    return pairs


def _train_hard_negative(
    parent: Path,
    output_path: Path,
    seed: int,
    masks: Mapping[str, Any],
    natural: Mapping[str, np.ndarray],
    tts: Mapping[str, np.ndarray],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    set_deterministic_cpu()
    schedule_payload = read_json(parent / "03_train" / str(seed) / "schedule.json")
    source_schedule = list(schedule_payload.get("entries", []))
    expected = TRAIN_STEPS * BATCH_SIZE
    if len(source_schedule) < expected:
        raise ValueError(f"parent schedule is shorter than the fixed 600-step budget for seed {seed}")
    schedule = source_schedule[:expected]
    base_checkpoint = parent / "03_train" / str(seed) / BASE_MODEL_ARM / "checkpoint.pt"
    _, base_payload = load_checkpoint(base_checkpoint)
    torch.manual_seed(int(seed))
    initial_model = MaskedNaturalReconstructor()
    initial_state = clone_state(initial_model.state_dict())
    if state_hash(initial_state) != base_payload["initial_state_sha256"]:
        raise ValueError(f"initial state does not match parent checkpoint for seed {seed}")
    model = MaskedNaturalReconstructor()
    model.load_state_dict(clone_state(initial_state), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    pairs = _build_train_pairs(masks, natural, tts, stats)
    trace: list[dict[str, float]] = []
    for step in range(TRAIN_STEPS):
        selected = [pairs[str(row["mask_sha256"])] for row in schedule[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]
        paired_batch = _batch_examples([item[0] for item in selected])
        wrong_batch = _batch_examples([item[1] for item in selected])
        optimizer.zero_grad(set_to_none=True)
        paired_prediction = model(paired_batch["natural_mel"], paired_batch["masked_support"], paired_batch["target_core"], paired_batch["tts_features"])
        wrong_prediction = model(wrong_batch["natural_mel"], wrong_batch["masked_support"], wrong_batch["target_core"], wrong_batch["tts_features"])
        paired_loss = reconstruction_loss(paired_prediction, paired_batch["target"], paired_batch["target_core"])
        wrong_loss = reconstruction_loss(wrong_prediction, wrong_batch["target"], wrong_batch["target_core"])
        ranking_penalty = torch.clamp_min(0.01 + paired_loss["total"] - wrong_loss["total"], 0.0)
        training_loss = paired_loss["total"] + 0.10 * ranking_penalty
        training_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        values = {
            "paired_patch": float(paired_loss["patch"].detach()),
            "paired_velocity": float(paired_loss["velocity"].detach()),
            "paired_total": float(paired_loss["total"].detach()),
            "wrong_total": float(wrong_loss["total"].detach()),
            "ranking_penalty": float(ranking_penalty.detach()),
            "training_total": float(training_loss.detach()),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"non-finite hard-negative training loss for seed {seed}")
        trace.append(values)
    payload = {
        "schema_version": 1,
        "seed": int(seed),
        "arm": "hard_negative_same_phone",
        "optimizer_step": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "schedule_sha256": schedule_hash(schedule),
        "parent_schedule_sha256": str(schedule_payload.get("sha256", "")),
        "initial_state_sha256": state_hash(initial_state),
        "model_config": model.config(),
        "objective": {"margin": 0.01, "ranking_weight": 0.10, "formula": "paired_total + 0.10 * max(0, 0.01 + paired_total - wrong_instance_total)"},
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP},
        "loss_first": trace[0],
        "loss_final": trace[-1],
        "loss_trace_sha256": sha256_text(canonical_json(trace)),
        "state_dict": model.state_dict(),
        "sealed_splits_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {key: value for key, value in payload.items() if key != "state_dict"}


def _evaluate_training_reconstruction(
    run_dir: Path,
    parent: Path,
    confirmation: Path,
    controls: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output = run_dir / "04_training/reconstruction"
    result_path = run_dir / "04_training/reconstruction.json"
    if result_path.exists() and resume:
        return read_json(result_path)
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    records = _cohort_records(cohort)
    _, masks, stats, natural, tts = _load_parent(parent)
    mask_by_hash = _mask_map(masks)
    feature_map = _load_npz(run_dir / "02_controls/features.npz")
    control_by_mask = {str(row["mask_sha256"]): row for row in controls["records"]}
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        model, payload = load_checkpoint(run_dir / "04_training" / str(seed) / "hard_negative" / "checkpoint.pt")
        for record in records:
            for mask_hash in record["evaluation_mask_sha256"]:
                target = mask_by_hash[str(mask_hash)]
                base = {key: np.asarray(value, dtype=np.float32) for key, value in build_example(target, natural, tts, stats).items() if key != "provenance"}
                control_row = control_by_mask[str(mask_hash)]
                examples = {
                    "PAIRED_TTS": base,
                    "SAME_PHONE_WRONG_INSTANCE": _control_examples(target, feature_map, control_row, natural, tts, stats, "SAME_PHONE_WRONG_INSTANCE"),
                    "WITHIN_PHONE_REVERSED": _control_examples(target, feature_map, control_row, natural, tts, stats, "WITHIN_PHONE_REVERSED"),
                    "NAT_ONLY": base,
                }
                predictions: dict[str, np.ndarray] = {}
                losses: dict[str, dict[str, float]] = {}
                for condition, example in examples.items():
                    prediction, loss = score_condition(model, example, zero_tts=condition == "NAT_ONLY")
                    predictions[condition] = prediction
                    losses[condition] = loss
                for condition in ("PAIRED_TTS",) + CONTROL_CONDITIONS + ("NAT_ONLY",):
                    cell = output / str(seed) / str(mask_hash) / condition
                    cell.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cell / "prediction.npz", prediction=predictions[condition])
                    row = {
                        "schema_version": 1,
                        "seed": int(seed),
                        "sample_id": str(target["sample_id"]),
                        "source_group": str(target["source_group"]),
                        "mask_sha256": str(mask_hash),
                        "condition": condition,
                        "status": "complete",
                        "loss": losses[condition],
                        "paired_loss": losses["PAIRED_TTS"],
                        "reconstruction_gain": float(losses[condition]["total"]) - float(losses["PAIRED_TTS"]["total"]),
                        "prediction_shape": list(predictions[condition].shape),
                        "prediction_sha256": _array_hash(predictions[condition]),
                        "checkpoint_sha256": sha256_file(run_dir / "04_training" / str(seed) / "hard_negative" / "checkpoint.pt"),
                        "checkpoint_initial_state_sha256": payload["initial_state_sha256"],
                    }
                    write_protocol_json(cell / "record.json", row, overwrite=True)
                    rows.append(row)
    expected = sum(len(row["evaluation_mask_sha256"]) for row in records) * len(SEEDS) * 4
    if len(rows) != expected:
        raise ValueError(f"hard-negative reconstruction matrix incomplete: {len(rows)}/{expected}")
    result = {
        "schema_version": 1,
        "status": "complete",
        "record_count": EXPECTED_RECORDS,
        "mask_count": expected // (len(SEEDS) * 4),
        "seed_count": len(SEEDS),
        "condition_count": 4,
        "required_cells": expected,
        "conditions": ["PAIRED_TTS", *CONTROL_CONDITIONS, "NAT_ONLY"],
        "records": rows,
        "sealed_splits_accessed": False,
    }
    write_protocol_json(result_path, result, overwrite=resume)
    return result


def _part_c_reconstruction_gate(
    run_dir: Path,
    parent: Path,
    records: Sequence[Mapping[str, Any]],
    reconstruction: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(reconstruction["records"])
    paired_rows = {(str(row["sample_id"]), int(row["seed"])): row for row in rows if row["condition"] == "PAIRED_TTS"}
    contrasts = {}
    for condition in CONTROL_CONDITIONS:
        local = [row for row in rows if row["condition"] == condition]
        contrasts[condition] = _aggregate_contrast(
            records,
            local,
            paired_rows,
            lambda row, paired: float(row["loss"]["total"]) - float(paired["loss"]["total"]),
            "reconstruction_gain",
        )
    new_pair = _aggregate_absolute(
        records,
        [row for row in rows if row["condition"] == "PAIRED_TTS"],
        lambda row: float(row["loss"]["total"]),
        "paired_total",
    )
    old = read_json(run_dir / "03_frozen_diagnosis/reconstruction.json")
    old_rows = [row for row in old["records"] if row["condition"] == CONTROL_CONDITIONS[0]]
    old_pair = _aggregate_absolute(
        records,
        old_rows,
        lambda row: float(row["paired_loss"]["total"]),
        "paired_total",
    )
    quality = bool(new_pair["summary"]["median"] <= 1.05 * old_pair["summary"]["median"])
    hard_controls_pass = all(contrasts[condition]["summary"]["pass"] for condition in CONTROL_CONDITIONS)
    return {
        "status": "PASS" if hard_controls_pass and quality else "FAIL",
        "hard_controls_pass": hard_controls_pass,
        "paired_quality_guard": {
            "pass": quality,
            "new_median": new_pair["summary"]["median"],
            "original_median": old_pair["summary"]["median"],
            "maximum_allowed": 1.05 * old_pair["summary"]["median"],
        },
        "contrasts": contrasts,
        "new_paired": new_pair,
        "original_paired": old_pair,
        "rules": {
            "hard_control": "95% CI lower bound > 0 and at least 7/8 positive source groups for both controls",
            "paired_quality": "new paired eight-group median <= 1.05 * original paired eight-group median",
        },
        "sealed_splits_accessed": False,
    }


def _build_training_drivers(
    run_dir: Path,
    parent: Path,
    confirmation: Path,
    reconstruction: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    output_dir = run_dir / "04_training/mel_drivers"
    manifest_path = output_dir / "drivers.json"
    if manifest_path.exists() and resume:
        return read_json(manifest_path)
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    records = _cohort_records(cohort)
    _, masks, stats, natural_mels, _ = _load_parent(parent)
    mask_by_hash = _mask_map(masks)
    recon_by_key = {(int(row["seed"]), str(row["mask_sha256"]), str(row["condition"])): row for row in reconstruction["records"]}
    conditions = ("PAIRED_TTS", *CONTROL_CONDITIONS, "NAT_ONLY")
    driver_rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        natural = np.asarray(natural_mels[sid], dtype=np.float32)
        mean = np.asarray(stats["mel_mean"], dtype=np.float32)[:, None]
        std = np.asarray(stats["mel_std"], dtype=np.float32)[:, None]
        natural_standardized = (natural - mean) / std
        for seed in SEEDS:
            contributions_by_condition = {condition: [] for condition in conditions}
            used_by_condition = {condition: [] for condition in conditions}
            for mask_hash in record["evaluation_mask_sha256"]:
                mask = mask_by_hash[str(mask_hash)]
                global_start = int(mask["window_start_frame"]) + int(mask["core_start"])
                global_end = int(mask["window_start_frame"]) + int(mask["core_end"])
                if global_start < 0 or global_end > natural.shape[1] or global_start >= global_end:
                    raise ValueError(f"training prediction core outside natural mel for {mask_hash}")
                for condition in conditions:
                    recon = recon_by_key[(int(seed), str(mask_hash), condition)]
                    prediction_path = run_dir / "04_training/reconstruction" / str(seed) / str(mask_hash) / condition / "prediction.npz"
                    with np.load(prediction_path, allow_pickle=False) as archive:
                        prediction = np.asarray(archive["prediction"], dtype=np.float32)
                    if _array_hash(prediction) != str(recon["prediction_sha256"]):
                        raise ValueError(f"training prediction hash changed: {prediction_path}")
                    core_start, core_end = int(mask["core_start"]), int(mask["core_end"])
                    contributions_by_condition[condition].append((global_start, global_end, prediction[core_start:core_end]))
                    used_by_condition[condition].append({
                        "mask_sha256": str(mask_hash),
                        "global_start_frame": global_start,
                        "global_end_frame": global_end,
                        "core_start": core_start,
                        "core_end": core_end,
                        "prediction_sha256": str(recon["prediction_sha256"]),
                    })
            for condition in conditions:
                candidate, counts = patch_normalized_mel(natural_standardized, contributions_by_condition[condition])
                denormalized = candidate * std + mean
                output = natural.copy()
                covered = counts > 0
                output[:, covered] = denormalized[:, covered]
                unclamped = output.copy()
                clamp_mask = (output < MEL_MIN) | (output > MEL_MAX)
                output = np.clip(output, MEL_MIN, MEL_MAX).astype(np.float32, copy=False)
                if np.any(output[:, ~covered] != natural[:, ~covered]):
                    raise ValueError(f"training driver changed outside-core frames for {sid} {condition}")
                driver_id = f"{sid}__{condition}__{seed}"
                path = output_dir / f"{driver_id}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, output, allow_pickle=False)
                driver_rows.append({
                    "driver_id": driver_id,
                    "sample_id": sid,
                    "source_group": str(record["source_group"]),
                    "condition": condition,
                    "seed": int(seed),
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "shape": list(output.shape),
                    "natural_mel_sha256": _array_hash(natural),
                    "clamp_fraction": float(np.mean(clamp_mask)),
                    "clamped_values": int(np.count_nonzero(clamp_mask)),
                    "total_values": int(output.size),
                    "mel_range_before_clamp": [float(unclamped.min()), float(unclamped.max())],
                    "mel_range_after_clamp": [float(output.min()), float(output.max())],
                    "covered_frame_count": int(np.count_nonzero(covered)),
                    "overlap_frame_count": int(np.count_nonzero(counts > 1)),
                    "used_masks": used_by_condition[condition],
                    "outside_core_policy": "exact_natural_mel_source",
                })
    expected = EXPECTED_RECORDS * len(SEEDS) * 4
    if len(driver_rows) != expected:
        raise ValueError(f"training driver matrix incomplete: {len(driver_rows)}/{expected}")
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_hard_negative_direct_mel_drivers",
        "status": "complete",
        "cohort_manifest_sha256": sha256_file(confirmation / "00_cohort/manifest.json"),
        "reconstruction_sha256": sha256_file(run_dir / "04_training/reconstruction.json"),
        "natural_mel_source_sha256": sha256_file(parent / "02_features/natural_mels.npz"),
        "normalization_sha256": sha256_file(parent / "02_features/normalization.json"),
        "conditions": list(conditions),
        "seeds": list(SEEDS),
        "record_count": EXPECTED_RECORDS,
        "drivers_per_record": len(SEEDS) * 4,
        "driver_count": len(driver_rows),
        "overlap_policy": "arithmetic_mean_of_contributing_prediction_cores",
        "outside_core_policy": "exact_natural_mel_source",
        "valid_mel_range": [MEL_MIN, MEL_MAX],
        "drivers": driver_rows,
        "sealed_splits_accessed": False,
    }
    write_protocol_json(manifest_path, manifest, overwrite=resume)
    return manifest


def _analyze_training_downstream(
    run_dir: Path,
    confirmation: Path,
    syncnet: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    analysis_path = run_dir / "04_training/analysis.json"
    if analysis_path.exists() and resume:
        return read_json(analysis_path)
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    records = _cohort_records(cohort)
    rows = list(syncnet.get("scores", []))
    if syncnet.get("status") != "complete" or len(rows) != EXPECTED_RECORDS * len(SEEDS) * 4:
        raise ValueError("hard-negative downstream matrix is incomplete")
    paired_rows = {(str(row["sample_id"]), int(row["seed"])): row for row in rows if row["condition"] == "PAIRED_TTS"}
    hard = {}
    for condition in CONTROL_CONDITIONS:
        local = [row for row in rows if row["condition"] == condition]
        hard[condition] = {
            "Sync-C": _aggregate_contrast(records, local, paired_rows, lambda row, paired: float(paired["sync_c"]) - float(row["sync_c"]), "sync_c_gain"),
            "Sync-D": _aggregate_contrast(records, local, paired_rows, lambda row, paired: float(row["sync_d"]) - float(paired["sync_d"]), "sync_d_gain"),
        }
    nat_rows = [row for row in rows if row["condition"] == "NAT_ONLY"]
    modality = {
        "Sync-C": _aggregate_contrast(records, nat_rows, paired_rows, lambda row, paired: float(paired["sync_c"]) - float(row["sync_c"]), "modality_C_gain"),
        "Sync-D": _aggregate_contrast(records, nat_rows, paired_rows, lambda row, paired: float(row["sync_d"]) - float(paired["sync_d"]), "modality_D_gain"),
    }
    hard_pass = all(hard[condition][metric]["summary"]["pass"] for condition in CONTROL_CONDITIONS for metric in ("Sync-C", "Sync-D"))
    modality_pass = all(modality[metric]["summary"]["pass"] for metric in ("Sync-C", "Sync-D"))
    status = "TRAINING_TFG_GO" if hard_pass and modality_pass else "TRAINING_NO_GO"
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "part": "C",
        "training_status": status,
        "hard_controls": hard,
        "modality_over_nat_only": modality,
        "rules": {
            "hard_controls": "paired over both controls passes for Sync-C and Sync-D",
            "modality": "paired over NAT_ONLY passes the existing modality rule",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "sealed_splits_accessed": False,
    }
    write_protocol_json(analysis_path, analysis, overwrite=resume)
    lines = ["# Part C: hard-negative downstream feasibility", "", f"Status: **{status}**.", "", "The fixed hard-negative model was evaluated with the frozen direct-mel Wav2Lip, strict untouched-natural-audio replacement, and official SyncNet V2.", "", "| Condition | Metric | Median gain | 95% CI | Positive groups | Pass |", "|---|---|---:|---|---:|---|"]
    for condition in CONTROL_CONDITIONS:
        for metric in ("Sync-C", "Sync-D"):
            item = hard[condition][metric]["summary"]
            lines.append(f"| {condition} | {metric} | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if item['pass'] else 'no'} |")
    for metric in ("Sync-C", "Sync-D"):
        item = modality[metric]["summary"]
        lines.append(f"| PAIRED_TTS over NAT_ONLY | {metric} | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if item['pass'] else 'no'} |")
    lines.extend(["", "## Claim boundary", "", *[f"- {item}" for item in CLAIM_BOUNDARY], ""])
    write_protocol_json(run_dir / "04_training/analysis.json", analysis, overwrite=resume)
    (run_dir / "04_training/report.md").write_text("\n".join(lines), encoding="utf-8")
    return analysis


def run_part_c(
    run_dir: Path,
    parent: Path,
    confirmation: Path,
    controls: Mapping[str, Any],
    part_b: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    trajectory_status = str(part_b.get("trajectory_status"))
    if trajectory_status == "TFG_TRAJECTORY_SIGNAL":
        return {"status": "SKIPPED_ALREADY_SENSITIVE"}
    if trajectory_status == "RECONSTRUCTION_TRAJECTORY_SIGNAL_ONLY":
        return {"status": "SKIPPED_ENDPOINT_LIMITED"}
    if trajectory_status == "NOT_EVALUATED":
        return {"status": "SKIPPED_INVALID_DIAGNOSIS"}
    if trajectory_status != "NO_TRAJECTORY_SIGNAL":
        return {"status": "NOT_EVALUATED", "reason": "unknown Part B status"}
    status_path = run_dir / "04_training/status.json"
    if status_path.exists() and resume:
        return read_json(status_path)
    training_root = run_dir / "04_training"
    training_root.mkdir(parents=True, exist_ok=True)
    lock, masks, stats, natural, tts = _load_parent(parent)
    del lock
    for seed in SEEDS:
        checkpoint_path = training_root / str(seed) / "hard_negative" / "checkpoint.pt"
        if checkpoint_path.is_file() and resume:
            continue
        _train_hard_negative(parent, checkpoint_path, int(seed), masks, natural, tts, stats)
        print(f"hard-negative seed {seed} complete", flush=True)
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    records = _cohort_records(cohort)
    reconstruction = _evaluate_training_reconstruction(run_dir, parent, confirmation, controls, resume=resume)
    gate = _part_c_reconstruction_gate(run_dir, parent, records, reconstruction)
    write_protocol_json(training_root / "reconstruction_gate.json", gate, overwrite=resume)
    if gate["status"] != "PASS":
        status = {"status": "TRAINING_NO_GO", "reconstruction_gate": gate, "sealed_splits_accessed": False}
        write_protocol_json(status_path, status, overwrite=resume)
        (training_root / "report.md").write_text("# Part C: hard-negative training feasibility\n\nStatus: **TRAINING_NO_GO**.\n\nThe fixed hard-negative model did not pass the pre-registered reconstruction gate; no downstream rendering was performed.\n", encoding="utf-8")
        return status
    drivers = _build_training_drivers(run_dir, parent, confirmation, reconstruction, resume=resume)
    render_path = training_root / "renders/render_manifest.json"
    if not render_path.exists() or not resume:
        render_controls_parallel(
            run_dir,
            confirmation / "00_cohort/manifest.json",
            training_root / "mel_drivers/drivers.json",
            confirmation / "02_renders/parity.json",
            confirmation / "02_renders/render_manifest.json",
            training_root / "renders",
        )
    if not (training_root / "05_syncnet/summary.json").exists() or not resume:
        score_controls_parallel(training_root, confirmation / "00_cohort/manifest.json", render_path, workers=4)
    syncnet = read_json(training_root / "05_syncnet/summary.json")
    analysis = _analyze_training_downstream(run_dir, confirmation, syncnet, resume=resume)
    status = {"status": analysis["training_status"], "reconstruction_gate": gate, "analysis_sha256": sha256_file(training_root / "analysis.json"), "driver_count": drivers["driver_count"], "render_count": int(read_json(render_path)["render_count"]), "score_count": int(syncnet["score_count"]), "sealed_splits_accessed": False}
    write_protocol_json(status_path, status, overwrite=resume)
    return status


def final_decision(run_dir: Path, part_a: Mapping[str, Any], part_b: Mapping[str, Any], part_c_status: str) -> dict[str, Any]:
    if part_b.get("trajectory_status") == "NOT_EVALUATED" or part_c_status == "NOT_EVALUATED":
        recommendation = "STOP_INVALID_EXPERIMENT"
    elif part_b.get("trajectory_status") == "TFG_TRAJECTORY_SIGNAL" or part_c_status == "TRAINING_TFG_GO":
        recommendation = "CONFIRM_TRAJECTORY_MODEL_ON_NEW_RECORDS"
    elif part_a.get("bootstrap", {}).get("centroid_modality_C", {}).get("pass") and part_a.get("bootstrap", {}).get("centroid_modality_D", {}).get("pass"):
        recommendation = "USE_SIMPLER_PHONE_CONTROL_PATH"
    elif part_a.get("engineering_audit") == "PAIRED_BEATS_NATURAL_REFERENCE":
        recommendation = "RETAIN_PAIRED_MODALITY_PATH"
    else:
        recommendation = "USE_NATURAL_REFERENCE_PATH"
    return {
        "schema_version": 1,
        "status": "complete",
        "recommendation": recommendation,
        "part_a_status": part_a.get("engineering_audit"),
        "part_b_status": part_b.get("trajectory_status"),
        "part_c_status": part_c_status,
        "previously_inspected_records": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "waveform_decoder_gate": "CLOSED",
        "sealed_splits_accessed": False,
    }


def execute(args: argparse.Namespace) -> int:
    parent = args.parent.resolve()
    confirmation = args.confirmation.resolve()
    run_dir = args.run.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise ValueError(f"refusing non-empty output directory without --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "binding", "reference", "controls", "reconstruction", "drivers", "render", "score", "analysis", "all"
    }
    if args.stage not in stages:
        raise ValueError(args.stage)
    binding = run_binding(run_dir, parent, confirmation, resume=args.resume)
    if args.stage == "binding":
        return 0
    cohort = read_json(confirmation / "00_cohort/manifest.json")
    part_a = _reference_audit(run_dir, confirmation, cohort, resume=args.resume)
    if args.stage == "reference":
        return 0
    controls = build_controls(run_dir, parent, confirmation, resume=args.resume)
    if args.stage == "controls":
        return 0
    reconstruction = run_reconstruction(run_dir, parent, confirmation, controls, resume=args.resume)
    if args.stage == "reconstruction":
        return 0
    drivers = build_control_drivers(run_dir, parent, confirmation, controls, reconstruction, resume=args.resume)
    if args.stage == "drivers":
        return 0
    render_manifest_path = run_dir / "03_frozen_diagnosis/renders/render_manifest.json"
    if not render_manifest_path.exists() or not args.resume:
        render_all(
            run_dir,
            confirmation / "00_cohort/manifest.json",
            run_dir / "03_frozen_diagnosis/mel_drivers/drivers.json",
            confirmation / "02_renders/parity.json",
            run_dir / "03_frozen_diagnosis/renders",
        )
    render_manifest = read_json(render_manifest_path)
    if args.stage == "render":
        return 0
    score_path = run_dir / "05_syncnet/summary.json"
    if not score_path.exists() or not args.resume:
        score_controls_parallel(run_dir, confirmation / "00_cohort/manifest.json", render_manifest_path)
    syncnet = read_json(score_path)
    if args.stage == "score":
        return 0
    part_b = analyze_frozen(run_dir, confirmation, controls, reconstruction, syncnet, resume=args.resume)
    part_c = run_part_c(run_dir, parent, confirmation, controls, part_b, resume=args.resume)
    part_c_status = str(part_c["status"])
    decision = final_decision(run_dir, part_a, part_b, part_c_status)
    write_protocol_json(run_dir / "decision.json", decision, overwrite=args.resume)
    summary = {
        "schema_version": 1,
        "run": "lrs3_masked_tts_trajectory_specificity_20260902",
        "status": "complete",
        "binding": {"manifest_sha256": sha256_file(run_dir / "00_binding/manifest.json")},
        "part_a": {"status": part_a["engineering_audit"], "analysis_sha256": sha256_file(run_dir / "01_reference_audit/analysis.json")},
        "part_b": {"status": part_b["trajectory_status"], "analysis_sha256": sha256_file(run_dir / "03_frozen_diagnosis/analysis.json"), "reconstruction_cells": reconstruction["required_cells"], "downstream_cells": syncnet["score_count"]},
        "part_c": {"status": part_c_status, "reconstruction_gate": part_c.get("reconstruction_gate"), "driver_count": part_c.get("driver_count"), "render_count": part_c.get("render_count"), "score_count": part_c.get("score_count")},
        "decision": decision,
        "sealed_splits_accessed": False,
    }
    write_protocol_json(run_dir / "summary.json", summary, overwrite=args.resume)
    print(json.dumps({"part_a": part_a["engineering_audit"], "part_b": part_b["trajectory_status"], "part_c": part_c_status, "recommendation": decision["recommendation"]}, ensure_ascii=False))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--confirmation", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("binding", "reference", "controls", "reconstruction", "drivers", "render", "score", "analysis", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
