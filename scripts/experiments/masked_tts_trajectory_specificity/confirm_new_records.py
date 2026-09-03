"""Confirm the trained trajectory model on an independent LRS3 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

import numpy as np
import soundfile as sf
import torch

from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import parse_textgrid
from scripts.experiments.masked_tts_reconstruction.config import (
    BATCH_SIZE,
    CPU_THREADS,
    MEL_BINS,
    SEEDS,
    WAVLM_DIM,
)
from scripts.experiments.masked_tts_reconstruction.evaluate import array_hash, score_condition
from scripts.experiments.masked_tts_reconstruction.features import (
    build_example,
    extract_natural_mel,
    phone_phase_linear,
    standardize_mel,
)
from scripts.experiments.masked_tts_reconstruction.model import MaskedNaturalReconstructor
from scripts.experiments.masked_tts_reconstruction.protocol import (
    build_mask_manifest,
    canonical_json,
    normalize_phone,
    sha256_file,
    sha256_text,
    write_json as write_protocol_json,
)
from scripts.experiments.masked_tts_reconstruction.run import _load_features, _load_mels
from scripts.experiments.masked_tts_reconstruction.train import load_checkpoint, set_deterministic_cpu
from scripts.experiments.masked_tts_tfg_probe.mel_drivers import MEL_MAX, MEL_MIN, patch_normalized_mel
from scripts.experiments.masked_tts_tfg_probe.run import (
    EXPECTED_SYNCNET_SHA256,
    EXPECTED_WAV2LIP_SHA256,
    MIN_TRACK,
    SYNCNET,
    SYNCNET_MODEL,
    SYNCNET_PY,
    WAV2LIP_CHECKPOINT,
    WAV2LIP_PY,
    _last_json,
    _parse_syncnet,
    _run_logged,
)
from scripts.experiments.masked_tts_trajectory_specificity.run import (
    _aggregate_contrast,
    _eligible_donors,
    _metric_summary,
    _reverse_inside_core,
    _select_donor,
    render_controls_parallel,
    score_controls_parallel,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3.mfa3_alignment import normalize_mfa3_transcript

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_RUN = REPO / "runs/lrs3_masked_tts_trajectory_specificity_20260902"
DEFAULT_FEATURE_RUN = REPO / "runs/lrs3_masked_tts_retention_exploratory_20260901"
DEFAULT_RUN = REPO / "runs/lrs3_masked_tts_new_confirmation_20260902"
CONDITIONS = ("PAIRED_TTS", "SAME_PHONE_WRONG_INSTANCE", "WITHIN_PHONE_REVERSED", "NAT_ONLY")
CONTROLS = CONDITIONS[1:3]
EXPECTED_RECORDS = 16
EXPECTED_GROUPS = 8
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260902


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_cohort(run_dir: Path) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    cohort = read_json(run_dir / "00_cohort/manifest.json")
    records = list(cohort.get("records", []))
    if cohort.get("status") != "complete" or len(records) != EXPECTED_RECORDS or len(cohort.get("groups", [])) != EXPECTED_GROUPS:
        raise ValueError("new confirmation cohort is incomplete")
    if len({str(row["sample_id"]) for row in records}) != EXPECTED_RECORDS:
        raise ValueError("new confirmation cohort has duplicate sample IDs")
    if any(sum(str(row["source_group"]) == str(group) for row in records) != 2 for group in cohort["groups"]):
        raise ValueError("new confirmation cohort does not contain two records per source group")
    return cohort, records


def prepare_mfa_inputs(run_dir: Path) -> dict[str, Any]:
    cohort, records = load_cohort(run_dir)
    tts = read_json(run_dir / "01_tts/tts_meta.json")
    if tts.get("complete") is not True or int(tts.get("samples_ok", -1)) != EXPECTED_RECORDS:
        raise ValueError("new Qwen TTS output is incomplete")
    tts_by_id = {str(key): value for key, value in tts.get("results", {}).items()}
    root = run_dir / "02_alignment"
    for side in ("natural", "tts"):
        target = root / "input" / side
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"refusing non-empty MFA input directory: {target}")
        target.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        sid = str(record["sample_id"])
        natural = Path(str(record["natural_audio"])).resolve()
        tts_row = tts_by_id.get(sid)
        if tts_row is None or tts_row.get("status") != "ok":
            raise ValueError(f"missing TTS result for {sid}")
        tts_audio = Path(str(tts_row["canonical_16k_audio"])).resolve()
        if not natural.is_file() or sha256_file(natural) != str(record["natural_audio_sha256"]):
            raise ValueError(f"natural audio hash mismatch for {sid}")
        if not tts_audio.is_file() or sha256_file(tts_audio) != str(tts_row["canonical_audio_sha256"]):
            raise ValueError(f"TTS audio hash mismatch for {sid}")
        lab_text = normalize_mfa3_transcript(str(record["transcript"]))
        for side, source in (("natural", natural), ("tts", tts_audio)):
            destination = root / "input" / side / f"{sid}.wav"
            shutil.copy2(source, destination)
            (root / "input" / side / f"{sid}.lab").write_text(lab_text + "\n", encoding="utf-8")
        rows.append({
            "sample_id": sid,
            "source_group": str(record["source_group"]),
            "natural_audio": str(natural),
            "natural_audio_sha256": sha256_file(natural),
            "tts_audio": str(tts_audio),
            "tts_audio_sha256": sha256_file(tts_audio),
            "transcript": str(record["transcript"]),
            "mfa_transcript": lab_text,
        })
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_new_confirmation_mfa_inputs",
        "status": "complete",
        "cohort_manifest_sha256": sha256_file(run_dir / "00_cohort/manifest.json"),
        "tts_manifest_sha256": sha256_file(run_dir / "01_tts/tts_meta.json"),
        "records": rows,
        "dictionary": "english_us_mfa",
        "acoustic_model": "english_mfa",
        "sealed_splits_accessed": False,
    }
    write_json(root / "input_manifest.json", manifest)
    return manifest


def run_mfa(run_dir: Path, mfa_bin: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "02_alignment/input_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("MFA input manifest is incomplete")
    root = run_dir / "02_alignment"
    command_rows = []
    for side in ("natural", "tts"):
        output = root / "output" / side
        output.mkdir(parents=True, exist_ok=True)
        command = [str(mfa_bin), "align", "--clean", "--overwrite", str(root / "input" / side), "english_us_mfa", "english_mfa", str(output), "--single_speaker", "--num_jobs", "4"]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "logs" / f"{side}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"MFA failed for {side}; see {root / 'logs' / f'{side}.log'}")
        command_rows.append({"side": side, "command": command, "returncode": int(result.returncode)})
    missing = []
    for row in manifest["records"]:
        for side in ("natural", "tts"):
            path = root / "output" / side / f"{row['sample_id']}.TextGrid"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise ValueError(f"MFA TextGrids missing: {missing[:5]}")
    result = {"schema_version": 1, "status": "complete", "mfa_bin": str(mfa_bin.resolve()), "commands": command_rows, "input_manifest_sha256": sha256_file(root / "input_manifest.json"), "sealed_splits_accessed": False}
    write_json(root / "mfa_run.json", result)
    return result


def build_alignment_records(run_dir: Path) -> dict[str, Any]:
    input_manifest = read_json(run_dir / "02_alignment/input_manifest.json")
    rows = []
    for item in input_manifest["records"]:
        sid = str(item["sample_id"])
        sides = {}
        for side in ("natural", "tts"):
            grid = run_dir / "02_alignment/output" / side / f"{sid}.TextGrid"
            parsed = parse_textgrid(grid)
            sides[f"{side}_phones"] = [{"phone": str(token["label"]), "start": float(token["start_s"]), "end": float(token["end_s"])} for token in parsed]
            sides[f"{side}_textgrid"] = str(grid.resolve())
            sides[f"{side}_textgrid_sha256"] = sha256_file(grid)
        rows.append({"sample_id": sid, "source_group": str(item["source_group"]), "transcript": str(item["transcript"]), **sides})
        write_json(run_dir / "02_alignment/records" / f"{sid}.json", rows[-1])
    result = {"schema_version": 1, "manifest_type": "lrs3_masked_tts_new_confirmation_phone_alignments", "status": "complete", "record_count": len(rows), "records": rows, "input_manifest_sha256": sha256_file(run_dir / "02_alignment/input_manifest.json"), "sealed_splits_accessed": False}
    write_json(run_dir / "02_alignment/alignment.json", result)
    return result


def extract_new_features(run_dir: Path, feature_run: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    _, records = load_cohort(run_dir)
    tts = read_json(run_dir / "01_tts/tts_meta.json")
    tts_by_id = {str(key): value for key, value in tts["results"].items()}
    natural: dict[str, np.ndarray] = {}
    tts_features: dict[str, np.ndarray] = {}
    feature_dir = run_dir / "03_data/tts_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    adapter = None
    for record in records:
        sid = str(record["sample_id"])
        natural[sid] = extract_natural_mel(Path(str(record["natural_audio"])))
        feature_path = feature_dir / f"{sid}.npy"
        if feature_path.is_file():
            features = np.asarray(np.load(feature_path, allow_pickle=False), dtype=np.float32)
        else:
            values, sample_rate = sf.read(str(tts_by_id[sid]["canonical_16k_audio"]), dtype="float32", always_2d=False)
            if int(sample_rate) != 16000 or np.asarray(values).ndim != 1:
                raise ValueError(f"invalid TTS audio for {sid}")
            if adapter is None:
                from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter
                source = Path(torch.hub.get_dir()) / "bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358"
                adapter = WavLMKNNVCAdapter.load_pretrained(device="cpu", source=source)
            features = adapter.extract(torch.from_numpy(np.asarray(values, dtype=np.float32))).cpu().numpy().astype(np.float32, copy=False)
            np.save(feature_path, features, allow_pickle=False)
        if features.ndim != 2 or features.shape[1] != WAVLM_DIM or features.shape[0] < 2 or not np.isfinite(features).all():
            raise ValueError(f"invalid new TTS WavLM-L6 features for {sid}")
        tts_features[sid] = np.ascontiguousarray(features, dtype=np.float32)
    stats = read_json(feature_run / "02_features/normalization.json")
    save_npz(run_dir / "03_data/natural_mels.npz", natural)
    write_json(run_dir / "03_data/feature_manifest.json", {
        "schema_version": 1,
        "status": "complete",
        "record_count": len(records),
        "natural_mel_shapes": {sid: list(value.shape) for sid, value in natural.items()},
        "tts_feature_shapes": {sid: list(value.shape) for sid, value in tts_features.items()},
        "normalization_source": str((feature_run / "02_features/normalization.json").resolve()),
        "normalization_sha256": sha256_file(feature_run / "02_features/normalization.json"),
        "sealed_splits_accessed": False,
    })
    return natural, tts_features, stats


def save_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(values[key], dtype=np.float32) for key in sorted(values)})


def load_parent_donor_pool(model_run: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    lock = read_json(model_run / "00_lock/lock.json")
    masks = read_json(model_run / "01_masks/mask_manifest.json")
    natural, tts, stats = _load_features(model_run, lock, masks)
    train_masks = [row for row in masks["masks"] if row.get("prototype_split") == "train"]
    if not train_masks:
        raise ValueError("parent train donor pool is empty")
    return lock, masks, natural, tts, stats


def build_new_masks(run_dir: Path, feature_run: Path, natural: Mapping[str, np.ndarray], tts: Mapping[str, np.ndarray]) -> dict[str, Any]:
    cohort, records = load_cohort(run_dir)
    alignment = read_json(run_dir / "02_alignment/alignment.json")
    alignment_by_id = {str(row["sample_id"]): row for row in alignment["records"]}
    lock_records = []
    for record in records:
        sid = str(record["sample_id"])
        alignment_row = alignment_by_id[sid]
        lock_records.append({
            "sample_id": sid,
            "source_group": str(record["source_group"]),
            "prototype_split": "evaluation",
            "paths": {"natural_audio": str(Path(str(record["natural_audio"])).resolve()), "tts_audio": str(Path(str(read_json(run_dir / '01_tts/tts_meta.json')["results"][sid]["canonical_16k_audio"])).resolve()), "alignment": str((run_dir / "02_alignment/records" / f"{sid}.json").resolve())},
        })
    lock = {"records": lock_records, "groups": {"order": list(cohort["groups"]), "evaluation": list(cohort["groups"])}}
    tts_lengths = {sid: int(value.shape[0]) for sid, value in tts.items()}
    raw = build_mask_manifest(lock, natural, tts_lengths, group_order=tuple(cohort["groups"]), evaluation_groups=tuple(cohort["groups"]), readiness_profile={"train_records_min": 0, "train_groups_min": 0, "train_masks_min": 0, "evaluation_records_min": EXPECTED_RECORDS, "evaluation_groups_exact": EXPECTED_GROUPS, "evaluation_masks_min": 1, "evaluation_masks_per_group_min": 1})
    _, parent_masks, _, parent_tts, _ = load_parent_donor_pool(feature_run)
    parent_train_masks = [row for row in parent_masks["masks"] if row.get("prototype_split") == "train"]
    eligible = []
    excluded = []
    for mask in raw["masks"]:
        donors = _eligible_donors(mask, parent_train_masks, parent_tts)
        if donors:
            eligible.append(mask)
        else:
            excluded.append({"mask_sha256": str(mask["mask_sha256"]), "sample_id": str(mask["sample_id"]), "reason": "no eligible parent-train same-phone donor"})
    if not eligible:
        raise ValueError("no new masks have eligible frozen donors")
    by_group = {group: sum(str(row["source_group"]) == str(group) for row in eligible) for group in cohort["groups"]}
    if any(value < 1 for value in by_group.values()):
        raise ValueError(f"new mask coverage lacks a source group: {by_group}")
    eligible.sort(key=lambda row: (tuple(cohort["groups"]).index(str(row["source_group"])), str(row["sample_id"]).encode("utf-8"), int(row["operation_index"]), int(row["natural_core_start_frame"])))
    for index, row in enumerate(eligible):
        row["canonical_index"] = index
    manifest = {**raw, "status": "complete", "masks": eligible, "counts": {"masks": len(eligible), "exclusions": len(raw.get("exclusions", [])) + len(excluded)}, "exclusions": [*raw.get("exclusions", []), *excluded], "record_count": EXPECTED_RECORDS, "group_count": EXPECTED_GROUPS, "groups": list(cohort["groups"]), "mask_order_sha256": sha256_text(canonical_json(eligible)), "parent_donor_pool": "frozen prior trajectory experiment prototype_split=train masks only", "sealed_splits_accessed": False}
    write_json(run_dir / "03_data/lock.json", lock)
    write_json(run_dir / "03_data/mask_manifest.json", manifest)
    return manifest


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {str(key): np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"non-finite feature cache: {path}")
    return result


def run_reconstruction(run_dir: Path, model_run: Path, feature_run: Path, masks: Mapping[str, Any], natural: Mapping[str, np.ndarray], new_tts: Mapping[str, np.ndarray], stats: Mapping[str, Any]) -> dict[str, Any]:
    _, parent_masks, parent_natural, parent_tts, _ = load_parent_donor_pool(feature_run)
    del parent_natural
    donor_pool = [row for row in parent_masks["masks"] if row.get("prototype_split") == "train"]
    all_tts = {**parent_tts, **new_tts}
    mask_by_hash = {str(row["mask_sha256"]): row for row in masks["masks"]}
    cohort, records = load_cohort(run_dir)
    del cohort
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        checkpoint = model_run / "04_training" / str(seed) / "hard_negative" / "checkpoint.pt"
        model, payload = load_checkpoint(checkpoint)
        for record in records:
            for mask_hash in record_masks(record, mask_by_hash):
                target = mask_by_hash[mask_hash]
                base = build_example(target, natural, new_tts, stats)
                donors = _eligible_donors(target, donor_pool, parent_tts)
                donor, donor_index = _select_donor(target, donors)
                wrong = build_example(target, natural, all_tts, stats, tts_mask=donor)
                reversed_features = _reverse_inside_core(np.asarray(base["tts_features"], dtype=np.float32), int(target["core_start"]), int(target["core_end"]))
                reversed_example = {key: np.asarray(base[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "target")}
                reversed_example["tts_features"] = reversed_features
                examples = {"PAIRED_TTS": base, "SAME_PHONE_WRONG_INSTANCE": wrong, "WITHIN_PHONE_REVERSED": reversed_example, "NAT_ONLY": base}
                predictions: dict[str, np.ndarray] = {}
                losses: dict[str, dict[str, float]] = {}
                for condition, example in examples.items():
                    prediction, loss = score_condition(model, example, zero_tts=condition == "NAT_ONLY")
                    predictions[condition] = prediction
                    losses[condition] = loss
                for condition in CONDITIONS:
                    cell = run_dir / "04_reconstruction" / str(seed) / mask_hash / condition
                    cell.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cell / "prediction.npz", prediction=predictions[condition])
                    rows.append({
                        "schema_version": 1,
                        "status": "complete",
                        "seed": int(seed),
                        "sample_id": str(target["sample_id"]),
                        "source_group": str(target["source_group"]),
                        "mask_sha256": mask_hash,
                        "condition": condition,
                        "loss": losses[condition],
                        "paired_loss": losses["PAIRED_TTS"],
                        "reconstruction_gain": float(losses[condition]["total"]) - float(losses["PAIRED_TTS"]["total"]),
                        "prediction_shape": list(predictions[condition].shape),
                        "prediction_sha256": array_hash(predictions[condition]),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "checkpoint_initial_state_sha256": payload["initial_state_sha256"],
                        "donor_source_mask_sha256": str(donor["mask_sha256"]),
                        "donor_source_sample_id": str(donor["sample_id"]),
                        "donor_selection_index": int(donor_index),
                        "donor_candidate_count": len(donors),
                    })
        print(f"new reconstruction seed {seed} complete", flush=True)
    expected = len(masks["masks"]) * len(SEEDS) * len(CONDITIONS)
    if len(rows) != expected:
        raise ValueError(f"new reconstruction matrix incomplete: {len(rows)}/{expected}")
    result = {"schema_version": 1, "manifest_type": "lrs3_masked_tts_new_confirmation_reconstruction", "status": "complete", "record_count": EXPECTED_RECORDS, "group_count": EXPECTED_GROUPS, "mask_count": len(masks["masks"]), "seed_count": len(SEEDS), "condition_count": len(CONDITIONS), "required_cells": expected, "conditions": list(CONDITIONS), "records": rows, "model_run": str(model_run.resolve()), "sealed_splits_accessed": False}
    write_json(run_dir / "04_reconstruction/reconstruction.json", result)
    return result


def record_masks(record: Mapping[str, Any], mask_by_hash: Mapping[str, Mapping[str, Any]]) -> list[str]:
    sid = str(record["sample_id"])
    values = [key for key, row in mask_by_hash.items() if str(row["sample_id"]) == sid]
    if not values:
        raise ValueError(f"new record has no usable evaluation masks: {sid}")
    return sorted(values, key=lambda key: int(mask_by_hash[key]["canonical_index"]))


def build_drivers(run_dir: Path, masks: Mapping[str, Any], natural: Mapping[str, np.ndarray], stats: Mapping[str, Any], reconstruction: Mapping[str, Any]) -> dict[str, Any]:
    _, records = load_cohort(run_dir)
    mask_by_hash = {str(row["mask_sha256"]): row for row in masks["masks"]}
    recon_by_key = {(int(row["seed"]), str(row["mask_sha256"]), str(row["condition"])): row for row in reconstruction["records"]}
    rows = []
    mean = np.asarray(stats["mel_mean"], dtype=np.float32)[:, None]
    std = np.asarray(stats["mel_std"], dtype=np.float32)[:, None]
    for record in records:
        sid = str(record["sample_id"])
        source = np.asarray(natural[sid], dtype=np.float32)
        standardized = (source - mean) / std
        for seed in SEEDS:
            contributions = {condition: [] for condition in CONDITIONS}
            used = {condition: [] for condition in CONDITIONS}
            for mask_hash in record_masks(record, mask_by_hash):
                mask = mask_by_hash[mask_hash]
                global_start = int(mask["window_start_frame"]) + int(mask["core_start"])
                global_end = int(mask["window_start_frame"]) + int(mask["core_end"])
                for condition in CONDITIONS:
                    recon = recon_by_key[(int(seed), mask_hash, condition)]
                    prediction_path = run_dir / "04_reconstruction" / str(seed) / mask_hash / condition / "prediction.npz"
                    with np.load(prediction_path, allow_pickle=False) as archive:
                        prediction = np.asarray(archive["prediction"], dtype=np.float32)
                    if array_hash(prediction) != str(recon["prediction_sha256"]):
                        raise ValueError(f"new reconstruction prediction hash changed: {prediction_path}")
                    core_start, core_end = int(mask["core_start"]), int(mask["core_end"])
                    contributions[condition].append((global_start, global_end, prediction[core_start:core_end]))
                    used[condition].append({"mask_sha256": mask_hash, "global_start_frame": global_start, "global_end_frame": global_end, "core_start": core_start, "core_end": core_end, "prediction_sha256": str(recon["prediction_sha256"])})
            for condition in CONDITIONS:
                candidate, counts = patch_normalized_mel(standardized, contributions[condition])
                output = source.copy()
                covered = counts > 0
                output[:, covered] = candidate[:, covered] * std + mean
                unclamped = output.copy()
                clamp_mask = (output < MEL_MIN) | (output > MEL_MAX)
                output = np.clip(output, MEL_MIN, MEL_MAX).astype(np.float32, copy=False)
                if np.any(output[:, ~covered] != source[:, ~covered]):
                    raise ValueError(f"new driver changed outside target cores: {sid} {condition}")
                path = run_dir / "05_drivers" / f"{sid}__{condition}__{seed}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, output, allow_pickle=False)
                rows.append({"driver_id": path.stem, "sample_id": sid, "source_group": str(record["source_group"]), "condition": condition, "seed": int(seed), "path": str(path.resolve()), "sha256": sha256_file(path), "shape": list(output.shape), "natural_mel_sha256": array_hash(source), "clamp_fraction": float(np.mean(clamp_mask)), "clamped_values": int(np.count_nonzero(clamp_mask)), "total_values": int(output.size), "mel_range_before_clamp": [float(unclamped.min()), float(unclamped.max())], "mel_range_after_clamp": [float(output.min()), float(output.max())], "covered_frame_count": int(np.count_nonzero(covered)), "overlap_frame_count": int(np.count_nonzero(counts > 1)), "used_masks": used[condition], "outside_core_policy": "exact_natural_mel_source"})
    expected = EXPECTED_RECORDS * len(SEEDS) * len(CONDITIONS)
    if len(rows) != expected:
        raise ValueError(f"new driver matrix incomplete: {len(rows)}/{expected}")
    result = {"schema_version": 1, "manifest_type": "lrs3_masked_tts_new_confirmation_direct_mel_drivers", "status": "complete", "record_count": EXPECTED_RECORDS, "group_count": EXPECTED_GROUPS, "driver_count": len(rows), "conditions": list(CONDITIONS), "seeds": list(SEEDS), "drivers": rows, "reconstruction_sha256": sha256_file(run_dir / "04_reconstruction/reconstruction.json"), "sealed_splits_accessed": False}
    write_json(run_dir / "05_drivers/drivers.json", result)
    return result


def make_parity(run_dir: Path, natural: Mapping[str, np.ndarray]) -> dict[str, Any]:
    _, records = load_cohort(run_dir)
    first = records[0]
    sid = str(first["sample_id"])
    mel_path = run_dir / "03_data/natural_mels" / f"{sid}.npy"
    mel_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_path, np.asarray(natural[sid], dtype=np.float32), allow_pickle=False)
    log = run_dir / "06_renders/parity.log"
    text = _run_logged([str(WAV2LIP_PY), str(REPO / "scripts/experiments/masked_tts_tfg_probe/direct_mel.py"), "parity", "--audio", str(first["natural_audio"]), "--mel", str(mel_path), "--fps", "25"], REPO, log)
    result = _last_json(text)
    result.update({"record": sid, "source_group": str(first["source_group"]), "driver_sha256": sha256_file(mel_path), "wav2lip_audio_sha256": sha256_file(REPO / "third_party/Wav2Lip/audio.py"), "wav2lip_hparams_sha256": sha256_file(REPO / "third_party/Wav2Lip/hparams.py")})
    write_json(run_dir / "06_renders/parity.json", result)
    return result


def build_box_manifest(run_dir: Path, natural: Mapping[str, np.ndarray]) -> dict[str, Any]:
    _, records = load_cohort(run_dir)
    rows = []
    for index, record in enumerate(records, 1):
        sid = str(record["sample_id"])
        mel_path = run_dir / "03_data/natural_mels" / f"{sid}.npy"
        if not mel_path.is_file():
            np.save(mel_path, np.asarray(natural[sid], dtype=np.float32), allow_pickle=False)
        boxes = run_dir / "06_renders/boxes" / f"{sid}.json"
        output = run_dir / "06_renders/box_probe" / f"{sid}.mp4"
        log = run_dir / "06_renders/box_probe_logs" / f"{sid}.log"
        if not boxes.is_file():
            command = [str(WAV2LIP_PY), str(REPO / "scripts/experiments/masked_tts_tfg_probe/direct_mel.py"), "render", "--checkpoint", str(WAV2LIP_CHECKPOINT), "--face", str(record["face"]), "--mel", str(mel_path), "--outfile", str(output), "--face-det-batch-size", "4", "--wav2lip-batch-size", "4", "--nosmooth", "--boxes-output", str(boxes)]
            _run_logged(command, REPO, log)
        parsed = json.loads(boxes.read_text(encoding="utf-8"))
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(f"invalid face boxes for {sid}")
        rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "boxes": str(boxes.resolve()), "box_count": len(parsed), "natural_probe_video": str(output.resolve()), "face_sha256": str(record["video_sha256"])})
        print(f"face boxes {index}/{len(records)} {sid}", flush=True)
    result = {"schema_version": 1, "status": "complete", "manifest_type": "lrs3_masked_tts_new_confirmation_face_boxes", "renders": rows, "sealed_splits_accessed": False}
    write_json(run_dir / "06_renders/box_manifest.json", result)
    return result


def analyze(run_dir: Path, reconstruction: Mapping[str, Any], syncnet: Mapping[str, Any]) -> dict[str, Any]:
    _, records = load_cohort(run_dir)
    recon_rows = list(reconstruction["records"])
    paired_recon = {(str(row["sample_id"]), int(row["seed"])): row for row in recon_rows if row["condition"] == "PAIRED_TTS"}
    recon = {}
    for condition in CONTROLS:
        local = [row for row in recon_rows if row["condition"] == condition]
        recon[condition] = _aggregate_contrast(records, local, paired_recon, lambda row, paired: float(row["loss"]["total"]) - float(paired["loss"]["total"]), "reconstruction_gain")
    score_rows = list(syncnet["scores"])
    paired_scores = {(str(row["sample_id"]), int(row["seed"])): row for row in score_rows if row["condition"] == "PAIRED_TTS"}
    downstream = {}
    for condition in CONTROLS:
        local = [row for row in score_rows if row["condition"] == condition]
        downstream[condition] = {"Sync-C": _aggregate_contrast(records, local, paired_scores, lambda row, paired: float(paired["sync_c"]) - float(row["sync_c"]), "sync_c_gain"), "Sync-D": _aggregate_contrast(records, local, paired_scores, lambda row, paired: float(row["sync_d"]) - float(paired["sync_d"]), "sync_d_gain")}
    natural_rows = [row for row in score_rows if row["condition"] == "NAT_ONLY"]
    modality = {"Sync-C": _aggregate_contrast(records, natural_rows, paired_scores, lambda row, paired: float(paired["sync_c"]) - float(row["sync_c"]), "modality_C_gain"), "Sync-D": _aggregate_contrast(records, natural_rows, paired_scores, lambda row, paired: float(row["sync_d"]) - float(paired["sync_d"]), "modality_D_gain")}
    recon_pass = all(recon[condition]["summary"]["pass"] for condition in CONTROLS)
    downstream_pass = all(downstream[condition][metric]["summary"]["pass"] for condition in CONTROLS for metric in ("Sync-C", "Sync-D"))
    modality_pass = all(modality[metric]["summary"]["pass"] for metric in ("Sync-C", "Sync-D"))
    status = "CONFIRMED_ON_NEW_RECORDS" if recon_pass and downstream_pass and modality_pass else "NOT_CONFIRMED_ON_NEW_RECORDS"
    result = {"schema_version": 1, "status": "complete", "confirmation_status": status, "record_count": EXPECTED_RECORDS, "group_count": EXPECTED_GROUPS, "mask_count": int(reconstruction["mask_count"]), "reconstruction": recon, "downstream": downstream, "modality_over_nat_only": modality, "rules": {"pass": "95% CI lower bound > 0 and at least 7/8 positive source groups", "aggregation": "mask median, seed median, record median, two-record source-group median", "audio": "untouched natural audio after strict replacement"}, "model_run": str(DEFAULT_MODEL_RUN.resolve()), "sealed_splits_accessed": False}
    write_json(run_dir / "08_analysis/analysis.json", result)
    lines = ["# Independent new-record confirmation", "", f"Status: **{status}**.", "", "| Contrast | Metric | Median | 95% CI | Positive groups | Pass |", "|---|---|---:|---|---:|---|"]
    for condition in CONTROLS:
        item = recon[condition]["summary"]
        lines.append(f"| reconstruction: paired over {condition} | total loss | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if item['pass'] else 'no'} |")
        for metric in ("Sync-C", "Sync-D"):
            item = downstream[condition][metric]["summary"]
            lines.append(f"| paired over {condition} | {metric} | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if item['pass'] else 'no'} |")
    for metric in ("Sync-C", "Sync-D"):
        item = modality[metric]["summary"]
        lines.append(f"| paired over no-TTS input | {metric} | {item['median']:.5f} | [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}] | {item['positive_groups']}/8 | {'yes' if item['pass'] else 'no'} |")
    (run_dir / "08_analysis/report.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "08_analysis/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def execute(args: argparse.Namespace) -> int:
    run_dir = args.run.resolve()
    model_run = args.model_run.resolve()
    feature_run = args.feature_run.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    stages = ("prepare", "align", "data", "reconstruction", "drivers", "render", "score", "analyze") if args.stage == "all" else (args.stage,)
    if "prepare" in stages:
        prepare_mfa_inputs(run_dir)
    if "align" in stages:
        run_mfa(run_dir, args.mfa_bin.resolve())
        build_alignment_records(run_dir)
    if "data" in stages:
        natural, new_tts, stats = extract_new_features(run_dir, feature_run)
        build_new_masks(run_dir, feature_run, natural, new_tts)
    else:
        natural = load_npz(run_dir / "03_data/natural_mels.npz") if (run_dir / "03_data/natural_mels.npz").is_file() else {}
        new_tts = {}
        stats = read_json(feature_run / "02_features/normalization.json") if any(stage in stages for stage in ("data", "reconstruction", "drivers")) else {}
    if "reconstruction" in stages:
        masks = read_json(run_dir / "03_data/mask_manifest.json")
        if not natural:
            natural = load_npz(run_dir / "03_data/natural_mels.npz")
        _, records = load_cohort(run_dir)
        for record in records:
            sid = str(record["sample_id"])
            feature_path = run_dir / "03_data/tts_features" / f"{sid}.npy"
            new_tts[sid] = np.asarray(np.load(feature_path, allow_pickle=False), dtype=np.float32)
        set_deterministic_cpu()
        reconstruction = run_reconstruction(run_dir, model_run, feature_run, masks, natural, new_tts, stats)
    else:
        reconstruction = read_json(run_dir / "04_reconstruction/reconstruction.json") if (run_dir / "04_reconstruction/reconstruction.json").is_file() else {}
    if "drivers" in stages:
        masks = read_json(run_dir / "03_data/mask_manifest.json")
        drivers = build_drivers(run_dir, masks, natural, stats, reconstruction)
    else:
        drivers = read_json(run_dir / "05_drivers/drivers.json") if (run_dir / "05_drivers/drivers.json").is_file() else {}
    if "render" in stages:
        if not natural:
            natural = load_npz(run_dir / "03_data/natural_mels.npz")
        make_parity(run_dir, natural)
        build_box_manifest(run_dir, natural)
        render_controls_parallel(run_dir, run_dir / "00_cohort/manifest.json", run_dir / "05_drivers/drivers.json", run_dir / "06_renders/parity.json", run_dir / "06_renders/box_manifest.json", run_dir / "06_renders", workers=args.render_workers)
    if "score" in stages:
        score_controls_parallel(run_dir, run_dir / "00_cohort/manifest.json", run_dir / "06_renders/render_manifest.json", workers=args.score_workers)
    if "analyze" in stages:
        syncnet = read_json(run_dir / "05_syncnet/summary.json")
        analysis = analyze(run_dir, reconstruction, syncnet)
        decision = {"schema_version": 1, "status": "complete", "recommendation": analysis["confirmation_status"], "waveform_decoder_gate": "CLOSED", "model_run": str(model_run), "analysis_sha256": sha256_file(run_dir / "08_analysis/analysis.json"), "sealed_splits_accessed": False}
        write_json(run_dir / "decision.json", decision)
        write_json(run_dir / "summary.json", {"schema_version": 1, "status": "complete", "decision": decision, "analysis": analysis, "sealed_splits_accessed": False})
        print(json.dumps({"confirmation_status": analysis["confirmation_status"]}, ensure_ascii=False))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model-run", type=Path, default=DEFAULT_MODEL_RUN)
    parser.add_argument("--feature-run", type=Path, default=DEFAULT_FEATURE_RUN)
    parser.add_argument("--mfa-bin", type=Path, default=Path.home() / ".venvs/mfa/bin/mfa")
    parser.add_argument("--stage", choices=("prepare", "align", "data", "reconstruction", "drivers", "render", "score", "analyze", "all"), default="all")
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--score-workers", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
