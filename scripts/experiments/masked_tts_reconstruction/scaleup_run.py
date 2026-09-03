"""Fixed fit-only scale-up for token-level TTS retention."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch

from .config import (
    BATCH_SIZE,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CPU_THREADS,
    EVAL_GROUPS as PROTOTYPE_EVAL_GROUPS,
    LEARNING_RATE,
    SEEDS,
    TRAIN_GROUPS as PROTOTYPE_TRAIN_GROUPS,
    VELOCITY_WEIGHT,
    WEIGHT_DECAY,
)
from .evaluate import array_hash, score_condition
from .features import (
    build_example,
    extract_natural_mel,
    feature_contract,
    fit_normalization,
    phone_phase_linear,
    standardize_tts,
)
from .model import MaskedNaturalReconstructor
from .protocol import (
    build_mask_manifest,
    canonical_json,
    leakage_audit,
    normalize_phone,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from .train import (
    clone_state,
    load_checkpoint,
    make_schedule,
    run_overfit_smoke,
    schedule_hash,
    set_deterministic_cpu,
    state_hash,
    train_arm,
)

RUN_NAME = "lrs3_masked_tts_retention_scaleup_20260901"
SELECTION_SALT = "masked-tts-retention-scaleup-v1\0"
SCALEUP_TRAIN_STEPS = 1_200
SCALEUP_BATCH_SIZE = BATCH_SIZE
MIN_TRAIN_RECORDS = 36
MIN_TRAIN_MASKS = 900
MIN_EVAL_RECORDS = 16
MIN_EVAL_MASKS = 400
MIN_EVAL_MASKS_PER_GROUP = 30
MIN_PHONE_INSTANCES = 20
MIN_PHONE_GROUPS = 3

MFA_TOKENS_PATH = "runs/lrs3_qwen_cloud_n500_20260817/03_rhythm_data_mfa_fixed_allow_unknown/mfa_tokens.json"
LOCAL_KNN_VC_SOURCE = Path(torch.hub.get_dir()) / "bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _code_hash() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_payload(train_groups: Sequence[str], evaluation_groups: Sequence[str]) -> dict[str, Any]:
    return {
        "selection_salt": SELECTION_SALT,
        "prototype_train_groups": list(PROTOTYPE_TRAIN_GROUPS),
        "prototype_evaluation_groups": list(PROTOTYPE_EVAL_GROUPS),
        "train_groups": list(train_groups),
        "evaluation_groups": list(evaluation_groups),
        "train_steps": SCALEUP_TRAIN_STEPS,
        "batch_size": SCALEUP_BATCH_SIZE,
        "seeds": list(SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "velocity_weight": VELOCITY_WEIGHT,
        "cpu_threads": CPU_THREADS,
        "phone_support": {"min_instances": MIN_PHONE_INSTANCES, "min_groups": MIN_PHONE_GROUPS},
    }


def _config_hash(train_groups: Sequence[str], evaluation_groups: Sequence[str]) -> str:
    return sha256_text(canonical_json(_config_payload(train_groups, evaluation_groups)))


def _source_pool_path(asset_root: Path) -> Path:
    return asset_root / "tmp/lrs3_policy_a1_200_20260828/source_pool.json"


def _mfa_tokens_path() -> Path:
    return _repo_root() / MFA_TOKENS_PATH


def _rank(value: str) -> str:
    return hashlib.sha256((SELECTION_SALT + value).encode("utf-8")).hexdigest()


def _load_fit_pool(asset_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = _source_pool_path(asset_root).resolve()
    pool = read_json(path)
    if pool.get("manifest_type") != "lrs3_replacement_source_pool" or pool.get("status") != "complete":
        raise ValueError("source pool is not the frozen complete LRS3 source pool")
    rows = pool.get("records")
    if not isinstance(rows, list):
        raise ValueError("source pool records are missing")
    fit_rows = [row for row in rows if isinstance(row, dict) and row.get("protocol_split") == "train"]
    if any(not row.get("sample_id") or not row.get("source_group") for row in fit_rows):
        raise ValueError("fit-only source pool contains an invalid record")
    by_id = {str(row["sample_id"]): row for row in fit_rows}
    if len(by_id) != len(fit_rows):
        raise ValueError("fit-only source pool contains duplicate sample IDs")
    token_path = _mfa_tokens_path().resolve()
    tokens = read_json(token_path)
    token_records = tokens.get("records")
    if not isinstance(token_records, dict):
        raise ValueError("MFA token cache records are missing")
    return path, by_id, token_records


def _group_plan(by_id: Mapping[str, Mapping[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...], list[dict[str, Any]]]:
    groups = sorted({str(row["source_group"]) for row in by_id.values()})
    prior_train = tuple(str(group) for group in PROTOTYPE_TRAIN_GROUPS)
    prior_eval = tuple(str(group) for group in PROTOTYPE_EVAL_GROUPS)
    if any(group not in groups for group in prior_train + prior_eval):
        raise ValueError("prior prototype group is absent from the fit-only source pool")
    fresh = sorted(set(groups) - set(prior_train) - set(prior_eval), key=_rank)
    candidates: list[dict[str, Any]] = []
    for group in fresh:
        candidates.append({"source_group": group, "rank_sha256": _rank(group), "fresh": True})
    train_groups = prior_train + tuple(fresh[:6])
    evaluation_groups = tuple(fresh[6:14])
    if len(fresh) < 14:
        return train_groups, evaluation_groups, candidates
    return train_groups, evaluation_groups, candidates


def _selected_rows(
    by_id: Mapping[str, Mapping[str, Any]],
    train_groups: Sequence[str],
    evaluation_groups: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_rows: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in by_id.values():
        group_rows[str(row["source_group"])].append(row)
    selected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for group in tuple(train_groups) + tuple(evaluation_groups):
        rows = sorted(group_rows.get(str(group), []), key=lambda row: _rank(str(row["sample_id"])))
        limit = 4 if group in train_groups else 3
        if len(rows) < 2:
            failures.append({"source_group": str(group), "reason": "fewer_than_two_records", "available": len(rows)})
        for row in rows[:limit]:
            selected.append({
                "sample_id": str(row["sample_id"]),
                "source_group": str(row["source_group"]),
                "parent_protocol_split": str(row["protocol_split"]),
                "prototype_split": "train" if group in train_groups else "evaluation",
                "selection_rank_sha256": _rank(str(row["sample_id"])),
                "source": dict(row),
            })
    selected.sort(key=lambda row: (
        tuple(train_groups).index(row["source_group"]) if row["source_group"] in train_groups else len(train_groups) + tuple(evaluation_groups).index(row["source_group"]),
        row["selection_rank_sha256"],
        row["sample_id"].encode("utf-8"),
    ))
    return selected, failures


def _token_alignment(sample_id: str, token_records: Mapping[str, Any]) -> dict[str, Any]:
    row = token_records.get(sample_id)
    if not isinstance(row, dict):
        raise FileNotFoundError(f"MFA token alignment is missing for {sample_id}")
    result: dict[str, Any] = {"record_id": sample_id}
    for side in ("natural", "tts"):
        source = row.get(f"{side}_tokens")
        if not isinstance(source, list) or not source:
            raise ValueError(f"MFA token alignment is invalid for {sample_id}: {side}")
        phones = []
        for token in source:
            if not isinstance(token, dict):
                raise ValueError(f"MFA token alignment has invalid interval for {sample_id}")
            label = token.get("token", token.get("canonical", ""))
            phones.append({
                "phone": str(label),
                "start": float(token["start_s"]),
                "end": float(token["end_s"]),
            })
        result[f"{side}_phones"] = phones
    return result


def _read_audio(path: Path) -> np.ndarray:
    values, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if int(sample_rate) != 16_000:
        raise ValueError(f"audio must be 16000 Hz: {path}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or values.size < 1024 or not np.isfinite(values).all():
        raise ValueError(f"audio must be finite mono data: {path}")
    return values


def _extract_wavlm_features(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    missing = []
    for row in rows:
        sid = str(row["sample_id"])
        target = output_dir / f"{sid}.npy"
        if target.is_file():
            values = np.asarray(np.load(target, allow_pickle=False))
            if values.ndim == 2 and values.shape[1] == 1024 and values.shape[0] >= 2 and np.isfinite(values).all():
                paths[sid] = target.resolve()
                continue
        missing.append((sid, Path(str(row["source"]["tts_audio"])).resolve(), target))
    model_meta: dict[str, Any] = {"repository": "bshall/knn-vc", "revision": "c616845c4e309e24d5927f15adbdf277a3d65358", "sample_rate": 16_000, "frame_stride_samples": 320, "selected_layer": 6, "feature_dim": 1024, "frozen": True}
    if missing:
        if not LOCAL_KNN_VC_SOURCE.is_dir():
            raise FileNotFoundError(f"local kNN-VC source is missing: {LOCAL_KNN_VC_SOURCE}")
        from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter

        adapter = WavLMKNNVCAdapter.load_pretrained(device="cpu", source=LOCAL_KNN_VC_SOURCE)
        model_meta.update(adapter.metadata())
        for sid, source_path, target in missing:
            values = _read_audio(source_path)
            features = adapter.extract(torch.from_numpy(values)).cpu().numpy().astype(np.float32, copy=False)
            if features.ndim != 2 or features.shape[1] != 1024 or features.shape[0] < 2 or not np.isfinite(features).all():
                raise ValueError(f"invalid WavLM-L6 output for {sid}")
            np.save(target, features)
            paths[sid] = target.resolve()
    return paths, model_meta


def _build_lock(run_dir: Path, asset_root: Path) -> dict[str, Any]:
    source_path, by_id, token_records = _load_fit_pool(asset_root)
    token_path = _mfa_tokens_path().resolve()
    train_groups, evaluation_groups, candidates = _group_plan(by_id)
    selected, selection_failures = _selected_rows(by_id, train_groups, evaluation_groups)
    if len(train_groups) != 12 or len(evaluation_groups) != 8:
        selection_failures.append({"reason": "group_denominator", "train_groups": len(train_groups), "evaluation_groups": len(evaluation_groups)})
    alignment_dir = run_dir / "00_lock/alignments"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row in selected:
        sid = str(row["sample_id"])
        alignment_path = alignment_dir / f"{sid}.json"
        alignment = _token_alignment(sid, token_records)
        if alignment_path.exists():
            if read_json(alignment_path) != alignment:
                raise ValueError(f"alignment snapshot changed for {sid}")
        else:
            write_json(alignment_path, alignment)
        source = row["source"]
        natural_audio = Path(str(source["natural_audio"])).resolve()
        tts_audio = Path(str(source["tts_audio"])).resolve()
        if not natural_audio.is_file() or not tts_audio.is_file():
            selection_failures.append({"sample_id": sid, "reason": "audio_missing"})
        records.append({**row, "paths": {"natural_audio": str(natural_audio), "tts_audio": str(tts_audio), "alignment": str(alignment_path.resolve())}})
    feature_paths, model_meta = _extract_wavlm_features(records, run_dir / "00_lock/tts_features")
    for row in records:
        sid = str(row["sample_id"])
        row["paths"]["tts_feature"] = str(feature_paths[sid])
        row["sha256"] = {
            "natural_audio": sha256_file(Path(row["paths"]["natural_audio"])),
            "tts_audio": sha256_file(Path(row["paths"]["tts_audio"])),
            "alignment": sha256_file(Path(row["paths"]["alignment"])),
            "tts_feature": sha256_file(Path(row["paths"]["tts_feature"])),
            "source_pool_row": sha256_text(canonical_json(row["source"])),
        }
        if row["sha256"]["natural_audio"] != str(row["source"].get("natural_audio_sha256")) or row["sha256"]["tts_audio"] != str(row["source"].get("tts_audio_sha256")):
            raise ValueError(f"source-pool audio hash mismatch for {sid}")
        row.pop("source", None)
    records_json = [{key: row[key] for key in ("sample_id", "source_group", "parent_protocol_split", "prototype_split", "selection_rank_sha256", "paths", "sha256")} for row in records]
    record_order_hash = sha256_text(canonical_json(records_json))
    group_counts = {group: sum(row["source_group"] == group for row in records) for group in tuple(train_groups) + tuple(evaluation_groups)}
    lock = {
        "schema_version": 1,
        "status": "GO" if not selection_failures else "INSUFFICIENT",
        "run": RUN_NAME,
        "asset_root": str(asset_root.resolve()),
        "metadata": {
            "source_pool": str(source_path),
            "source_pool_sha256": sha256_file(source_path),
            "mfa_tokens": str(token_path),
            "mfa_tokens_sha256": sha256_file(token_path),
        },
        "config": _config_payload(train_groups, evaluation_groups),
        "config_sha256": _config_hash(train_groups, evaluation_groups),
        "code_sha256": _code_hash(),
        "groups": {
            "train": list(train_groups),
            "evaluation": list(evaluation_groups),
            "order": list(tuple(train_groups) + tuple(evaluation_groups)),
            "prototype_train_excluded_from_fresh": list(PROTOTYPE_TRAIN_GROUPS),
            "prototype_evaluation_excluded": list(PROTOTYPE_EVAL_GROUPS),
            "fresh_candidates": candidates,
        },
        "records": records_json,
        "record_order_sha256": record_order_hash,
        "counts": {"records": len(records), "groups": len(set(row["source_group"] for row in records)), "train_records": sum(row["prototype_split"] == "train" for row in records), "evaluation_records": sum(row["prototype_split"] == "evaluation" for row in records), "group_counts": group_counts},
        "selection_failures": selection_failures,
        "tts_feature_model": model_meta,
        "sealed_splits_accessed": False,
        "score_blind": True,
    }
    write_json(run_dir / "00_lock/lock.json", lock)
    return lock


def _load_mels(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {str(key): np.asarray(archive[key], dtype=np.float32) for key in archive.files}


def _save_mels(path: Path, mels: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: np.asarray(mels[key], dtype=np.float32) for key in sorted(mels, key=lambda value: value.encode("utf-8"))}
    np.savez_compressed(path, **ordered)


def _load_tts(lock: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for row in lock["records"]:
        sid = str(row["sample_id"])
        values = np.asarray(np.load(Path(str(row["paths"]["tts_feature"])), allow_pickle=False), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 1024 or values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError(f"invalid locked TTS feature {sid}")
        if sha256_file(Path(str(row["paths"]["tts_feature"]))) != row["sha256"]["tts_feature"]:
            raise ValueError(f"locked TTS feature hash changed for {sid}")
        result[sid] = values
    return result


def _apply_phone_support(manifest: Mapping[str, Any]) -> dict[str, Any]:
    masks = [dict(row) for row in manifest["masks"]]
    counts: collections.Counter[str] = collections.Counter()
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for row in masks:
        if row["prototype_split"] == "train":
            label = normalize_phone(row["label"])
            counts[label] += 1
            groups[label].add(str(row["source_group"]))
    supported = sorted(label for label in counts if counts[label] >= MIN_PHONE_INSTANCES and len(groups[label]) >= MIN_PHONE_GROUPS)
    supported_set = set(supported)
    removed = [row for row in masks if normalize_phone(row["label"]) not in supported_set]
    kept = [row for row in masks if normalize_phone(row["label"]) in supported_set]
    for row in removed:
        manifest_reason = {"sample_id": row["sample_id"], "source_group": row["source_group"], "prototype_split": row["prototype_split"], "operation_index": row["operation_index"], "natural_phone_index": row["natural_phone_index"], "tts_phone_index": row["tts_phone_index"], "label": row["label"], "reason": "unsupported_phone"}
        manifest_reason["support_instances"] = int(counts[normalize_phone(row["label"])])
        manifest_reason["support_groups"] = int(len(groups[normalize_phone(row["label"])]))
        manifest_reason["mask_sha256"] = row["mask_sha256"]
        manifest.setdefault("exclusions", []).append(manifest_reason)
    for index, row in enumerate(kept):
        row["canonical_index"] = index
    result = dict(manifest)
    result["masks"] = kept
    result["counts"] = {"masks": len(kept), "exclusions": len(result.get("exclusions", []))}
    result["mask_order_sha256"] = sha256_text(canonical_json(kept))
    result["phone_support"] = {
        "min_instances": MIN_PHONE_INSTANCES,
        "min_groups": MIN_PHONE_GROUPS,
        "support_instances": {label: int(counts[label]) for label in sorted(counts)},
        "support_groups": {label: sorted(groups[label]) for label in sorted(groups)},
        "retained_labels": supported,
        "removed_mask_count": len(removed),
    }
    return result


def _scaleup_readiness(masks: Mapping[str, Any], train_groups: Sequence[str], evaluation_groups: Sequence[str]) -> dict[str, Any]:
    rows = list(masks["masks"])
    train = [row for row in rows if row["prototype_split"] == "train"]
    evaluation = [row for row in rows if row["prototype_split"] == "evaluation"]
    by_group = {group: sum(row["source_group"] == group for row in evaluation) for group in evaluation_groups}
    readiness = {
        "train_records": len({row["sample_id"] for row in train}),
        "train_groups": len({row["source_group"] for row in train}),
        "train_masks": len(train),
        "evaluation_records": len({row["sample_id"] for row in evaluation}),
        "evaluation_groups": len({row["source_group"] for row in evaluation}),
        "evaluation_masks": len(evaluation),
        "evaluation_masks_by_group": by_group,
    }
    readiness["sufficient"] = bool(
        readiness["train_records"] >= MIN_TRAIN_RECORDS
        and readiness["train_groups"] == len(tuple(train_groups)) == 12
        and readiness["train_masks"] >= MIN_TRAIN_MASKS
        and readiness["evaluation_records"] >= MIN_EVAL_RECORDS
        and readiness["evaluation_groups"] == len(tuple(evaluation_groups)) == 8
        and readiness["evaluation_masks"] >= MIN_EVAL_MASKS
        and all(value >= MIN_EVAL_MASKS_PER_GROUP for value in by_group.values())
    )
    return readiness


def _write_alignment_provenance(run_dir: Path, masks: Mapping[str, Any], tts: Mapping[str, np.ndarray]) -> None:
    path = run_dir / "01_masks/alignment_provenance.jsonl"
    if path.exists():
        return
    lines = []
    for mask in masks["masks"]:
        sid = str(mask["sample_id"])
        _, frames = phone_phase_linear(tts[sid], int(mask["tts_frame_start"]), int(mask["tts_frame_end"]), int(mask["core_end"]) - int(mask["core_start"]))
        lines.append(json.dumps({"mask_sha256": mask["mask_sha256"], "sample_id": sid, "source_mask_sha256": mask["mask_sha256"], "natural_phone_start_s": mask["natural_start_s"], "natural_phone_end_s": mask["natural_end_s"], "tts_phone_start_s": mask["tts_start_s"], "tts_phone_end_s": mask["tts_end_s"], "frames": frames}, ensure_ascii=False, sort_keys=True, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_masks(run_dir: Path, lock: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "01_masks/mask_manifest.json"
    if path.exists() and resume:
        return read_json(path)
    natural_path = run_dir / "02_features/natural_mels.npz"
    if natural_path.exists() and resume:
        natural = _load_mels(natural_path)
    else:
        natural = {str(row["sample_id"]): extract_natural_mel(Path(str(row["paths"]["natural_audio"]))) for row in lock["records"]}
        _save_mels(natural_path, natural)
    tts = _load_tts(lock)
    train_groups = tuple(lock["groups"]["train"])
    evaluation_groups = tuple(lock["groups"]["evaluation"])
    raw = build_mask_manifest(
        lock,
        natural,
        {sid: int(values.shape[0]) for sid, values in tts.items()},
        group_order=tuple(lock["groups"]["order"]),
        evaluation_groups=evaluation_groups,
        readiness_profile={
            "train_records_min": MIN_TRAIN_RECORDS,
            "train_groups_min": 12,
            "train_groups_exact": 12,
            "train_masks_min": MIN_TRAIN_MASKS,
            "evaluation_records_min": MIN_EVAL_RECORDS,
            "evaluation_groups_exact": 8,
            "evaluation_masks_min": MIN_EVAL_MASKS,
            "evaluation_masks_per_group_min": MIN_EVAL_MASKS_PER_GROUP,
        },
    )
    manifest = _apply_phone_support(raw)
    manifest["readiness"] = _scaleup_readiness(manifest, train_groups, evaluation_groups)
    manifest["train_groups"] = list(train_groups)
    manifest["evaluation_groups"] = list(evaluation_groups)
    manifest["source_pool_sha256"] = lock["metadata"]["source_pool_sha256"]
    manifest["normalization_scope"] = "frozen_train_groups_only"
    write_json(run_dir / "01_masks/primary_mask_manifest.json", manifest)
    write_json(path, manifest)
    write_json(run_dir / "01_masks/exclusion_ledger.json", {"schema_version": 1, "exclusions": manifest["exclusions"], "counts": manifest["counts"], "phone_support": manifest["phone_support"]})
    _write_alignment_provenance(run_dir, manifest, tts)
    return manifest


def _build_phone_centroids(masks: Mapping[str, Any], tts: Mapping[str, np.ndarray], stats: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frames: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    groups: dict[str, set[str]] = collections.defaultdict(set)
    records: dict[str, set[str]] = collections.defaultdict(set)
    mask_ids: dict[str, set[str]] = collections.defaultdict(set)
    for mask in masks["masks"]:
        if mask["prototype_split"] != "train":
            continue
        label = normalize_phone(mask["label"])
        values = standardize_tts(tts[str(mask["sample_id"])], stats)
        aligned, _ = phone_phase_linear(values, int(mask["tts_frame_start"]), int(mask["tts_frame_end"]), int(mask["core_end"]) - int(mask["core_start"]))
        frames[label].append(aligned)
        groups[label].add(str(mask["source_group"]))
        records[label].add(str(mask["sample_id"]))
        mask_ids[label].add(str(mask["mask_sha256"]))
    centroids: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for label in sorted(frames):
        values = np.concatenate(frames[label], axis=0).astype(np.float32, copy=False)
        if len(mask_ids[label]) < MIN_PHONE_INSTANCES or len(groups[label]) < MIN_PHONE_GROUPS:
            raise ValueError(f"unsupported phone survived manifest filter: {label}")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        if mean.shape != (1024,) or not np.isfinite(mean).all():
            raise ValueError(f"invalid centroid for phone {label}")
        centroids[label] = mean
        provenance[label] = {
            "label": label,
            "contributing_groups": sorted(groups[label]),
            "contributing_records": sorted(records[label]),
            "contributing_masks": sorted(mask_ids[label]),
            "frame_count": int(values.shape[0]),
            "mean_sha256": array_hash(mean),
        }
    if not centroids:
        raise ValueError("phone centroid table is empty")
    return centroids, {
        "schema_version": 1,
        "source": "frozen_train_groups_only",
        "train_manifest_sha256": str(masks["mask_order_sha256"]),
        "phone_count": len(centroids),
        "phones": provenance,
    }


def _save_centroids(run_dir: Path, centroids: Mapping[str, np.ndarray], provenance: Mapping[str, Any]) -> str:
    labels = np.asarray(sorted(centroids), dtype="<U32")
    values = np.stack([np.asarray(centroids[label], dtype=np.float32) for label in labels], axis=0)
    path = run_dir / "02_features/phone_centroids.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["labels"], labels) or not np.array_equal(archive["centroids"], values):
                raise ValueError("phone centroid artifact changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=labels, centroids=values)
    write_json(run_dir / "02_features/phone_centroids.json", provenance)
    return sha256_file(path)


def _load_centroids(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    path = run_dir / "02_features/phone_centroids.npz"
    with np.load(path, allow_pickle=False) as archive:
        labels = [str(label) for label in archive["labels"].tolist()]
        values = np.asarray(archive["centroids"], dtype=np.float32)
    if values.shape != (len(labels), 1024) or any(not np.isfinite(row).all() for row in values):
        raise ValueError("invalid phone centroid table")
    provenance = read_json(run_dir / "02_features/phone_centroids.json")
    return {label: values[index] for index, label in enumerate(labels)}, provenance, sha256_file(path)


def _run_features(run_dir: Path, lock: Mapping[str, Any], masks: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    stats_path = run_dir / "02_features/normalization.json"
    if stats_path.exists() and resume:
        stats = read_json(stats_path)
        if not (run_dir / "02_features/phone_centroids.npz").exists():
            centroids, provenance = _build_phone_centroids(masks, _load_tts(lock), stats)
            _save_centroids(run_dir, centroids, provenance)
        return stats
    natural = _load_mels(run_dir / "02_features/natural_mels.npz")
    tts = _load_tts(lock)
    train_ids = {str(row["sample_id"]) for row in lock["records"] if row["prototype_split"] == "train"}
    stats = fit_normalization(natural, tts, train_ids)
    write_json(stats_path, stats)
    write_json(run_dir / "02_features/contract.json", feature_contract(lock))
    write_json(run_dir / "02_features/feature_hashes.json", {row["sample_id"]: row["sha256"]["tts_feature"] for row in lock["records"]})
    centroids, provenance = _build_phone_centroids(masks, tts, stats)
    _save_centroids(run_dir, centroids, provenance)
    return stats


def _examples(lock: Mapping[str, Any], masks: Mapping[str, Any], natural: Mapping[str, np.ndarray], tts: Mapping[str, np.ndarray], stats: Mapping[str, Any], centroids: Mapping[str, np.ndarray], *, split: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    paired: dict[str, dict[str, np.ndarray]] = {}
    centroid_examples: dict[str, dict[str, np.ndarray]] = {}
    for mask in masks["masks"]:
        if mask["prototype_split"] != split:
            continue
        key = str(mask["mask_sha256"])
        built = build_example(mask, natural, tts, stats)
        pair = {field: np.asarray(built[field], dtype=np.float32) for field in ("natural_mel", "masked_support", "target_core", "tts_features", "target")}
        paired[key] = pair
        label = normalize_phone(mask["label"])
        if label not in centroids:
            raise ValueError(f"missing phone centroid for {label}")
        centroid = dict(pair)
        centroid["tts_features"] = np.zeros_like(pair["tts_features"])
        centroid["tts_features"][int(mask["core_start"]):int(mask["core_end"])] = centroids[label][None, :]
        centroid_examples[key] = centroid
    return paired, centroid_examples


def _training_binding(lock: Mapping[str, Any], masks: Mapping[str, Any], stats: Mapping[str, Any], centroid_sha256: str) -> dict[str, Any]:
    data_hash = sha256_text(canonical_json({"record_order_sha256": lock["record_order_sha256"], "mask_order_sha256": masks["mask_order_sha256"], "record_hashes": {row["sample_id"]: row["sha256"] for row in lock["records"]}, "centroid_sha256": centroid_sha256}))
    return {
        "code_sha256": _code_hash(),
        "config_sha256": lock["config_sha256"],
        "record_order_sha256": lock["record_order_sha256"],
        "mask_order_sha256": masks["mask_order_sha256"],
        "normalization_sha256": stats["sha256"],
        "centroid_sha256": centroid_sha256,
        "data_sha256": data_hash,
        "sealed_splits_accessed": False,
    }


def _checkpoint_metadata(path: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    model, payload = load_checkpoint(path)
    del model
    if payload.get("binding") != dict(binding) or payload.get("optimizer_step") != SCALEUP_TRAIN_STEPS:
        raise ValueError(f"checkpoint binding mismatch: {path}")
    return {key: value for key, value in payload.items() if key != "state_dict"}


def _run_train(run_dir: Path, lock: Mapping[str, Any], masks: Mapping[str, Any], stats: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    summary_path = run_dir / "03_train/training.json"
    centroids, _, centroid_sha256 = _load_centroids(run_dir)
    binding = _training_binding(lock, masks, stats, centroid_sha256)
    if summary_path.exists() and resume:
        summary = read_json(summary_path)
        if summary.get("binding") != binding or summary.get("arm_count") != 9:
            raise ValueError("existing scale-up training summary is not hash-valid")
        return summary
    natural = _load_mels(run_dir / "02_features/natural_mels.npz")
    tts = _load_tts(lock)
    paired, centroid_examples = _examples(lock, masks, natural, tts, stats, centroids, split="train")
    leakage = leakage_audit(next(iter(paired.values())))
    write_json(run_dir / "03_train/leakage_audit.json", leakage)
    smoke = run_overfit_smoke(paired, [], seed=SEEDS[0])
    write_json(run_dir / "03_train/overfit_smoke.json", smoke)
    set_deterministic_cpu()
    seed_results: dict[str, Any] = {}
    for seed in SEEDS:
        torch.manual_seed(int(seed))
        initial = MaskedNaturalReconstructor()
        initial_state = clone_state(initial.state_dict())
        initial_hash = state_hash(initial_state)
        schedule = make_schedule(masks, int(seed), train_groups=tuple(lock["groups"]["train"]), train_steps=SCALEUP_TRAIN_STEPS, batch_size=SCALEUP_BATCH_SIZE)
        schedule_sha256 = schedule_hash(schedule)
        seed_dir = run_dir / "03_train" / str(seed)
        schedule_path = seed_dir / "schedule.json"
        schedule_payload = {"schema_version": 1, "seed": int(seed), "steps": SCALEUP_TRAIN_STEPS, "batch_size": SCALEUP_BATCH_SIZE, "entries": schedule, "sha256": schedule_sha256}
        if schedule_path.exists():
            if read_json(schedule_path) != schedule_payload:
                raise ValueError(f"schedule changed for seed {seed}")
        else:
            write_json(schedule_path, schedule_payload)
        arm_results: dict[str, Any] = {}
        for arm, arm_examples in (("paired_tts", paired), ("phone_centroid", centroid_examples), ("nat_only", paired)):
            checkpoint = seed_dir / arm / "checkpoint.pt"
            if checkpoint.exists():
                arm_results[arm] = _checkpoint_metadata(checkpoint, binding)
                continue
            arm_results[arm] = train_arm(
                arm_examples,
                schedule,
                seed=int(seed),
                arm=arm,
                initial_state=initial_state,
                output_path=checkpoint,
                binding=binding,
                steps=SCALEUP_TRAIN_STEPS,
                batch_size=SCALEUP_BATCH_SIZE,
            )
        seed_results[str(seed)] = {"seed": int(seed), "initial_state_sha256": initial_hash, "schedule_sha256": schedule_sha256, "arms": arm_results, "thread_config": {"device": "cpu", "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": True}}
    training = {"schema_version": 1, "status": "GO", "steps": SCALEUP_TRAIN_STEPS, "batch_size": SCALEUP_BATCH_SIZE, "seeds": seed_results, "arm_names": ["paired_tts", "phone_centroid", "nat_only"], "arm_count": 9, "binding": binding, "overfit_smoke": smoke, "sealed_splits_accessed": False}
    write_json(summary_path, training)
    return training


def _run_evaluate(run_dir: Path, lock: Mapping[str, Any], masks: Mapping[str, Any], stats: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "04_eval/evaluation.json"
    if path.exists() and resume:
        return read_json(path)
    centroids, _, centroid_sha256 = _load_centroids(run_dir)
    natural = _load_mels(run_dir / "02_features/natural_mels.npz")
    tts = _load_tts(lock)
    paired, centroid_examples = _examples(lock, masks, natural, tts, stats, centroids, split="evaluation")
    evaluation_masks = [row for row in masks["masks"] if row["prototype_split"] == "evaluation"]
    evaluation_masks.sort(key=lambda row: int(row["canonical_index"]))
    rows: list[dict[str, Any]] = []
    conditions = (("PAIRED_TTS", "paired_tts", paired, False), ("PHONE_CENTROID", "phone_centroid", centroid_examples, False), ("NAT_ONLY", "nat_only", paired, True))
    for seed in SEEDS:
        checkpoints = {arm: run_dir / "03_train" / str(seed) / arm / "checkpoint.pt" for arm in ("paired_tts", "phone_centroid", "nat_only")}
        models = {arm: load_checkpoint(path)[0] for arm, path in checkpoints.items()}
        for mask in evaluation_masks:
            key = str(mask["mask_sha256"])
            for condition, arm, source_examples, zero_tts in conditions:
                example = source_examples[key]
                prediction, loss = score_condition(models[arm], example, zero_tts=zero_tts)
                cell_dir = run_dir / "04_eval" / str(seed) / key / condition
                cell_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cell_dir / "prediction.npz", prediction=prediction)
                record = {
                    "schema_version": 1,
                    "seed": int(seed),
                    "sample_id": str(mask["sample_id"]),
                    "source_group": str(mask["source_group"]),
                    "mask_sha256": key,
                    "condition": condition,
                    "status": "complete",
                    "loss": loss,
                    "prediction_shape": list(prediction.shape),
                    "prediction_sha256": array_hash(prediction),
                    "target_sha256": array_hash(example["target"]),
                    "natural_input_sha256": array_hash(example["natural_mel"]),
                    "masked_support_sha256": array_hash(example["masked_support"]),
                    "target_core_sha256": array_hash(example["target_core"]),
                    "tts_input_sha256": array_hash(np.zeros_like(example["tts_features"]) if zero_tts else example["tts_features"]),
                    "checkpoint_sha256": sha256_file(checkpoints[arm]),
                    "centroid_sha256": centroid_sha256,
                }
                write_json(cell_dir / "record.json", record)
                rows.append(record)
    expected = len(evaluation_masks) * len(SEEDS) * 3
    if len(rows) != expected:
        raise ValueError(f"scale-up evaluation expected {expected} cells, found {len(rows)}")
    result = {"schema_version": 1, "status": "GO", "evaluation_masks": len(evaluation_masks), "seeds": list(SEEDS), "required_cells": expected, "records": rows, "conditions": ["PAIRED_TTS", "PHONE_CENTROID", "NAT_ONLY"], "sealed_splits_accessed": False}
    write_json(path, result)
    return result


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires values")
    value = float(np.median(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(value):
        raise FloatingPointError("median is non-finite")
    return value


def _bootstrap(values: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    result = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        result[index] = np.median(values[rng.integers(0, len(values), size=len(values))])
    return {"median": float(np.median(values)), "ci95": [float(np.quantile(result, 0.025, method="linear")), float(np.quantile(result, 0.975, method="linear"))], "draws": int(draws), "seed": int(seed), "unit": "whole_source_group", "method": "numpy_quantile_linear"}


def _analyze(
    evaluation: Mapping[str, Any],
    masks: Mapping[str, Any],
    run_dir: Path,
    *,
    exploratory: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_groups = tuple(str(group) for group in masks["evaluation_groups"])
    required = {"PAIRED_TTS", "PHONE_CENTROID", "NAT_ONLY"}
    cells: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = collections.defaultdict(dict)
    for row in evaluation["records"]:
        if row.get("status") != "complete":
            continue
        key = (int(row["seed"]), str(row["mask_sha256"]))
        condition = str(row["condition"])
        if condition in required:
            if condition in cells[key]:
                raise ValueError(f"duplicate scale-up cell: {key} {condition}")
            cells[key][condition] = row
    eval_masks = {str(row["mask_sha256"]): row for row in masks["masks"] if row["prototype_split"] == "evaluation"}
    expected_keys = {(int(seed), key) for seed in evaluation["seeds"] for key in eval_masks}
    if set(cells) != expected_keys or any(set(value) != required for value in cells.values()):
        raise ValueError("scale-up evaluation cells are incomplete")
    mask_rows: list[dict[str, Any]] = []
    for (seed, key), by_condition in sorted(cells.items()):
        base = by_condition["PAIRED_TTS"]
        for condition in required:
            other = by_condition[condition]
            for field in ("sample_id", "source_group", "target_sha256", "natural_input_sha256", "masked_support_sha256", "target_core_sha256"):
                if other.get(field) != base.get(field):
                    raise ValueError(f"shared input mismatch for {key}: {field}")
        full = float(by_condition["PAIRED_TTS"]["loss"]["total"])
        centroid = float(by_condition["PHONE_CENTROID"]["loss"]["total"])
        natural = float(by_condition["NAT_ONLY"]["loss"]["total"])
        if not all(math.isfinite(value) for value in (full, centroid, natural)):
            raise FloatingPointError(f"non-finite scale-up loss for {key}")
        mask = eval_masks[key]
        mask_rows.append({"seed": seed, "mask_sha256": key, "sample_id": str(mask["sample_id"]), "source_group": str(mask["source_group"]), "paired_tts": full, "phone_centroid": centroid, "nat_only": natural, "modality_gain": natural - full, "token_gain": centroid - full})
    by_group_seed: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in mask_rows:
        by_group_seed[(row["source_group"], row["seed"])].append(row)
    record_rows: list[dict[str, Any]] = []
    group_seed_rows: list[dict[str, Any]] = []
    for (group, seed), group_rows in sorted(by_group_seed.items()):
        by_record: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in group_rows:
            by_record[row["sample_id"]].append(row)
        local_records = []
        for sample_id, rows in sorted(by_record.items(), key=lambda item: item[0].encode("utf-8")):
            record = {"source_group": group, "seed": seed, "sample_id": sample_id, "mask_count": len(rows), **{key: _median([float(row[key]) for row in rows]) for key in ("paired_tts", "phone_centroid", "nat_only", "modality_gain", "token_gain")}}
            record_rows.append(record)
            local_records.append(record)
        group_seed_rows.append({"source_group": group, "seed": seed, "record_count": len(local_records), "mask_count": len(group_rows), **{key: _median([float(row[key]) for row in local_records]) for key in ("paired_tts", "phone_centroid", "nat_only", "modality_gain", "token_gain")}})
    final_group_rows = []
    for group in eval_groups:
        rows = [row for row in group_seed_rows if row["source_group"] == group]
        if len(rows) != len(SEEDS):
            raise ValueError(f"evaluation group lacks three seeds: {group}")
        final_group_rows.append({"source_group": group, "seed_count": len(rows), **{key: _median([float(row[key]) for row in rows]) for key in ("paired_tts", "phone_centroid", "nat_only", "modality_gain", "token_gain")}})
    seed_overall = {}
    for seed in SEEDS:
        rows = [row for row in group_seed_rows if int(row["seed"]) == int(seed)]
        seed_overall[str(seed)] = {"seed": int(seed), "group_count": len(rows), "overall_median_modality_gain": _median([float(row["modality_gain"]) for row in rows]), "overall_median_token_gain": _median([float(row["token_gain"]) for row in rows])}
    modality_values = np.asarray([float(row["modality_gain"]) for row in final_group_rows], dtype=np.float64)
    token_values = np.asarray([float(row["token_gain"]) for row in final_group_rows], dtype=np.float64)
    modality_bootstrap = _bootstrap(modality_values, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    token_bootstrap = _bootstrap(token_values, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    median_nat = _median([float(row["nat_only"]) for row in final_group_rows])
    median_full = _median([float(row["paired_tts"]) for row in final_group_rows])
    median_centroid = _median([float(row["phone_centroid"]) for row in final_group_rows])
    modality_reduction = (median_nat - median_full) / median_nat
    token_reduction = (median_centroid - median_full) / median_centroid
    modality_gates = {
        "bootstrap_lower_bound_gt_zero": modality_bootstrap["ci95"][0] > 0,
        "at_least_7_of_8_groups_positive": int(np.sum(modality_values > 0)) >= 7,
        "all_seeds_overall_median_positive": all(float(row["overall_median_modality_gain"]) > 0 for row in seed_overall.values()),
        "median_relative_reduction_ge_5_percent": modality_reduction >= 0.05,
    }
    token_gates = {
        "bootstrap_lower_bound_gt_zero": token_bootstrap["ci95"][0] > 0,
        "at_least_7_of_8_groups_positive": int(np.sum(token_values > 0)) >= 7,
        "all_seeds_overall_median_positive": all(float(row["overall_median_token_gain"]) > 0 for row in seed_overall.values()),
        "median_relative_reduction_ge_2_percent": token_reduction >= 0.02,
    }
    readiness = dict(masks["readiness"])
    modality_pass = all(modality_gates.values())
    token_pass = modality_pass and all(token_gates.values())
    if exploratory:
        science = "EXPLORATORY_DESCRIPTIVE_ONLY"
    elif not readiness.get("sufficient", False):
        science = "INSUFFICIENT"
    elif token_pass:
        science = "TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED"
    elif modality_pass:
        science = "PHONE_LEVEL_ONLY_SUPPORTED"
    else:
        science = "NO_SCALEUP_SUPPORT"
    analysis = {
        "schema_version": 1,
        "status": "GO",
        "exploratory": bool(exploratory),
        "mask_rows": mask_rows,
        "record_rows": record_rows,
        "group_seed_rows": group_seed_rows,
        "final_group_rows": final_group_rows,
        "seed_overall": seed_overall,
        "bootstrap": {"modality_gain": modality_bootstrap, "token_gain": token_bootstrap},
        "relative_reduction": {"median_nat_only": median_nat, "median_paired_tts": median_full, "median_phone_centroid": median_centroid, "versus_nat_only": modality_reduction, "versus_phone_centroid": token_reduction},
        "gates": {"modality": modality_gates, "token": token_gates},
        "readiness": readiness,
        "sealed_splits_accessed": False,
    }
    decision = {
        "schema_version": 1,
        "engineering": "GO",
        "science": science,
        "exploratory": bool(exploratory),
        "failed_gates": [f"modality.{name}" for name, value in modality_gates.items() if not value] + [f"token.{name}" for name, value in token_gates.items() if not value],
        "primary": "modality_gain = L_total(NAT_ONLY) - L_total(PAIRED_TTS); token_gain = L_total(PHONE_CENTROID) - L_total(PAIRED_TTS)",
        "claim": "descriptive exploratory modality/token contrasts only; the frozen readiness denominator was not met" if exploratory else ("paired token-level TTS-L6 improves held-out masked natural-mel prediction beyond a train-only phone centroid under this frozen fit-only task" if science == "TOKEN_LEVEL_TTS_SIGNAL_SUPPORTED" else "no token-level TTS signal claim"),
        "claim_boundary": ["not perceptible TTS-feature retention", "not waveform reachability", "not natural-prosody preservation", "not audio quality", "not TFG/SyncNet gain", "not replacement effect", "not population generalization"],
        "gates": {"modality": modality_gates, "token": token_gates},
        "sealed_splits_accessed": False,
    }
    output = run_dir / "05_analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis.json", analysis, overwrite=True)
    write_json(output / "decision.json", decision, overwrite=True)
    return analysis, decision


def _validate(
    run_dir: Path,
    lock: Mapping[str, Any],
    masks: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    exploratory: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if lock.get("sealed_splits_accessed") is not False or masks.get("sealed_splits_accessed", False) is not False or training.get("sealed_splits_accessed") is not False or evaluation.get("sealed_splits_accessed") is not False:
        errors.append("sealed_splits_accessed is not false")
    if len(lock.get("groups", {}).get("train", [])) != 12 or len(lock.get("groups", {}).get("evaluation", [])) != 8:
        errors.append("frozen group counts are not 12/8")
    if masks.get("readiness", {}).get("sufficient") is not True and not exploratory:
        errors.append("scale-up readiness denominator is insufficient")
    if training.get("arm_count") != 9:
        errors.append("nine fixed training arms are not present")
    expected = len([row for row in masks["masks"] if row["prototype_split"] == "evaluation"]) * len(SEEDS) * 3
    if evaluation.get("required_cells") != expected:
        errors.append("required evaluation cell count mismatch")
    validation = {"schema_version": 1, "status": "GO" if not errors else "NO_GO", "errors": errors, "sealed_splits_accessed": False, "decision_science": decision.get("science")}
    write_json(run_dir / "validation.json", validation, overwrite=True)
    if errors:
        raise ValueError(f"scale-up artifact validation failed: {errors}")
    return validation



EXPLORATORY_RUN_NAME = "lrs3_masked_tts_retention_exploratory_20260901"


def _copy_if_hash_valid(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise ValueError(f"exploratory snapshot changed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != sha256_file(source):
        raise ValueError(f"exploratory snapshot hash mismatch: {target}")


def run_exploratory(options: argparse.Namespace) -> int:
    source_run = Path(options.exploratory_source_run).resolve()
    run_dir = Path(options.run).resolve()
    if source_run == run_dir:
        raise ValueError("exploratory source and output runs must differ")
    if run_dir.exists() and any(run_dir.iterdir()) and not options.resume:
        raise ValueError(f"refusing non-empty output directory without --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    source_lock_path = source_run / "00_lock/lock.json"
    if not source_lock_path.is_file():
        raise FileNotFoundError(source_lock_path)
    lock_path = run_dir / "00_lock/lock.json"
    if lock_path.exists() and options.resume:
        lock = read_json(lock_path)
    else:
        source_lock = read_json(source_lock_path)
        lock = dict(source_lock)
        lock["run"] = EXPLORATORY_RUN_NAME
        lock["execution_mode"] = "exploratory_descriptive"
        lock["source_run"] = str(source_run)
        lock["source_lock_sha256"] = sha256_file(source_lock_path)
        lock["code_sha256"] = _code_hash()
        write_json(lock_path, lock)
    for relative in (
        "01_masks/primary_mask_manifest.json",
        "01_masks/mask_manifest.json",
        "01_masks/exclusion_ledger.json",
        "01_masks/alignment_provenance.jsonl",
        "02_features/natural_mels.npz",
        "02_features/normalization.json",
        "02_features/contract.json",
        "02_features/feature_hashes.json",
        "02_features/phone_centroids.npz",
        "02_features/phone_centroids.json",
    ):
        _copy_if_hash_valid(source_run / relative, run_dir / relative)
    masks = read_json(run_dir / "01_masks/mask_manifest.json")
    stats = read_json(run_dir / "02_features/normalization.json")
    if not masks.get("readiness", {}).get("sufficient", False):
        mode = "exploratory_descriptive_only"
    else:
        mode = "exploratory"
    training = _run_train(run_dir, lock, masks, stats, resume=bool(options.resume))
    evaluation = _run_evaluate(run_dir, lock, masks, stats, resume=bool(options.resume))
    analysis, decision = _analyze(evaluation, masks, run_dir, exploratory=True)
    decision["execution_mode"] = mode
    write_json(run_dir / "05_analysis/decision.json", decision, overwrite=True)
    write_json(run_dir / "summary.json", {"schema_version": 1, "run": EXPLORATORY_RUN_NAME, "execution_mode": mode, "source_run": str(source_run), "lock_counts": lock["counts"], "mask_counts": masks["counts"], "readiness": masks["readiness"], "training": {"arm_count": training["arm_count"], "steps": training["steps"]}, "evaluation": {"required_cells": evaluation["required_cells"]}, "analysis": {"modality_gain_median": analysis["bootstrap"]["modality_gain"]["median"], "modality_ci95": analysis["bootstrap"]["modality_gain"]["ci95"], "token_gain_median": analysis["bootstrap"]["token_gain"]["median"], "token_ci95": analysis["bootstrap"]["token_gain"]["ci95"]}, "decision": decision, "sealed_splits_accessed": False}, overwrite=True)
    _validate(run_dir, lock, masks, training, evaluation, decision, exploratory=True)
    return 0


def run(options: argparse.Namespace) -> int:
    if options.exploratory:
        if options.exploratory_source_run is None:
            raise ValueError("--exploratory-source-run is required in exploratory mode")
        return run_exploratory(options)
    run_dir = Path(options.run).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not options.resume:
        raise ValueError(f"refusing non-empty output directory without --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "00_lock/lock.json"
    lock = read_json(lock_path) if lock_path.exists() and options.resume else _build_lock(run_dir, Path(options.asset_root).resolve())
    if options.stage == "lock":
        return 0
    masks = _run_masks(run_dir, lock, resume=bool(options.resume))
    if options.stage == "masks":
        return 0
    stats = _run_features(run_dir, lock, masks, resume=bool(options.resume))
    if options.stage == "features":
        return 0
    if not masks.get("readiness", {}).get("sufficient", False):
        decision = {
            "schema_version": 1,
            "engineering": "GO",
            "science": "INSUFFICIENT",
            "failed_gates": ["readiness.train_masks_min", "readiness.evaluation_masks_min"],
            "readiness": masks["readiness"],
            "claim": "no scientific contrast emitted because the frozen denominator is insufficient",
            "sealed_splits_accessed": False,
        }
        write_json(run_dir / "decision.json", decision, overwrite=bool(options.resume))
        write_json(run_dir / "summary.json", {"schema_version": 1, "run": RUN_NAME, "lock_counts": lock["counts"], "mask_counts": masks["counts"], "readiness": masks["readiness"], "training": {"started": False, "reason": "INSUFFICIENT"}, "evaluation": {"started": False, "reason": "INSUFFICIENT"}, "decision": decision, "sealed_splits_accessed": False}, overwrite=bool(options.resume))
        write_json(run_dir / "validation.json", {"schema_version": 1, "status": "GO", "errors": [], "completed_stages": ["lock", "masks", "features"], "training_started": False, "evaluation_started": False, "sealed_splits_accessed": False, "decision_science": "INSUFFICIENT"}, overwrite=bool(options.resume))
        return 0
    training = _run_train(run_dir, lock, masks, stats, resume=bool(options.resume))
    if options.stage == "train":
        return 0
    evaluation = _run_evaluate(run_dir, lock, masks, stats, resume=bool(options.resume))
    if options.stage == "evaluate":
        return 0
    analysis, decision = _analyze(evaluation, masks, run_dir)
    write_json(run_dir / "summary.json", {"schema_version": 1, "run": RUN_NAME, "lock_counts": lock["counts"], "mask_counts": masks["counts"], "readiness": masks["readiness"], "training": {"arm_count": training["arm_count"], "steps": training["steps"]}, "evaluation": {"required_cells": evaluation["required_cells"]}, "analysis": {"modality_gain_median": analysis["bootstrap"]["modality_gain"]["median"], "modality_ci95": analysis["bootstrap"]["modality_gain"]["ci95"], "token_gain_median": analysis["bootstrap"]["token_gain"]["median"], "token_ci95": analysis["bootstrap"]["token_gain"]["ci95"]}, "decision": decision, "sealed_splits_accessed": False}, overwrite=bool(options.resume))
    _validate(run_dir, lock, masks, training, evaluation, decision)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=_repo_root() / "runs" / RUN_NAME)
    parser.add_argument("--stage", choices=("lock", "masks", "features", "train", "evaluate", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--exploratory-source-run", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
