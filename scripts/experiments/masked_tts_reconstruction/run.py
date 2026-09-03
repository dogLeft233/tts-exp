"""CLI runner for the narrow masked reconstruction prototype."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .analyze import analyze_evaluation
from .config import EVAL_GROUPS, SEEDS, TRAIN_GROUPS, default_asset_root, frozen_constants
from .features import (
    build_example,
    extract_selected_features,
    feature_contract,
    fit_normalization,
    load_tts_feature,
    phone_phase_linear,
)
from .protocol import (
    assign_shuffled_donors,
    build_lock,
    build_mask_manifest,
    canonical_json,
    leakage_audit,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from .train import run_overfit_smoke, train_all
from .evaluate import evaluate_all

RUN_NAME = "lrs3_masked_tts_reconstruction_prototype_20260901"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_hash() -> str:
    return sha256_text(canonical_json(frozen_constants()))


def _prepare_run(run_dir: Path, resume: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise ValueError(f"refusing non-empty output directory without --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)


def _save_mels(path: Path, mels: Mapping[str, np.ndarray]) -> None:
    ordered = {key: np.asarray(mels[key], dtype=np.float32) for key in sorted(mels, key=lambda value: value.encode("utf-8"))}
    if any(value.dtype == object or not np.isfinite(value).all() for value in ordered.values()):
        raise ValueError("natural mel cache contains invalid values")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ordered)


def _load_mels(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {str(key): np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    if any(value.dtype == object or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("natural mel cache contains invalid values")
    return result


def _load_lock(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "00_lock/lock.json")


def _load_mask_manifest(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "01_masks/mask_manifest.json")


def _load_features(run_dir: Path, lock: Mapping[str, Any], masks: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    natural = _load_mels(run_dir / "02_features/natural_mels.npz")
    tts = {}
    for row in lock["records"]:
        sid = str(row["sample_id"])
        values = np.asarray(np.load(Path(str(row["paths"]["tts_feature"])), allow_pickle=False), dtype=np.float32)
        if values.dtype == object or not np.isfinite(values).all():
            raise ValueError(f"invalid TTS feature array {sid}")
        tts[sid] = values
    stats_path = run_dir / "02_features/normalization.json"
    stats = read_json(stats_path) if stats_path.exists() else {}
    return natural, tts, stats


def _build_examples(lock: Mapping[str, Any], masks: Mapping[str, Any], natural: Mapping[str, np.ndarray], tts: Mapping[str, np.ndarray], stats: Mapping[str, Any], *, split: str) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Mapping[str, Any]]]:
    rows = [row for row in masks["masks"] if row["prototype_split"] == split]
    by_hash = {str(row["mask_sha256"]): row for row in masks["masks"]}
    examples: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    for mask in rows:
        donor = None
        assignment = mask.get("shuffled_donor", {})
        if assignment.get("status") == "assigned":
            donor = by_hash.get(str(assignment["donor_mask_sha256"]))
            if donor is None:
                raise ValueError(f"shuffled donor missing from manifest for {mask['mask_sha256']}")
        example = build_example(mask, natural, tts, stats)
        examples[str(mask["mask_sha256"])] = {key: np.asarray(example[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "tts_features", "target")}
        metadata[str(mask["mask_sha256"])] = mask
        if split == "evaluation" and donor is not None:
            shuffled = build_example(mask, natural, tts, stats, tts_mask=donor)
            metadata[str(mask["mask_sha256"]) + ":shuffled"] = mask
            examples[str(mask["mask_sha256"]) + ":shuffled"] = {key: np.asarray(shuffled[key], dtype=np.float32) for key in ("natural_mel", "masked_support", "target_core", "tts_features", "target")}
    return examples, metadata


def run_lock(run_dir: Path, asset_root: Path) -> dict[str, Any]:
    path = run_dir / "00_lock/lock.json"
    if path.exists():
        return read_json(path)
    lock = build_lock(asset_root)
    lock["code_sha256"] = _code_hash()
    lock["config_sha256"] = _config_hash()
    write_json(path, lock)
    return lock


def _write_alignment_provenance(run_dir: Path, masks: Mapping[str, Any], tts: Mapping[str, np.ndarray]) -> None:
    path = run_dir / "01_masks/alignment_provenance.jsonl"
    if path.exists():
        return
    lines = []
    for mask in masks["masks"]:
        source = tts[str(mask["sample_id"])]
        _, frames = phone_phase_linear(source, int(mask["tts_frame_start"]), int(mask["tts_frame_end"]), int(mask["core_end"]) - int(mask["core_start"]))
        payload = {
            "mask_sha256": mask["mask_sha256"],
            "sample_id": mask["sample_id"],
            "natural_core_frame_centers_s": [float(index * 0.0125) for index in range(int(mask["natural_core_start_frame"]), int(mask["natural_core_end_frame"]))],
            "natural_phone_start_s": mask["natural_start_s"],
            "natural_phone_end_s": mask["natural_end_s"],
            "tts_phone_start_s": mask["tts_start_s"],
            "tts_phone_end_s": mask["tts_end_s"],
            "tts_frame_start": mask["tts_frame_start"],
            "tts_frame_end": mask["tts_frame_end"],
            "frames": frames,
        }
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_masks(run_dir: Path, lock: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    manifest_path = run_dir / "01_masks/mask_manifest.json"
    if manifest_path.exists() and resume:
        diagnostic = read_json(manifest_path)
        if "readiness" in diagnostic:
            return diagnostic
        primary = read_json(run_dir / "01_masks/primary_mask_manifest.json")
        diagnostic = assign_shuffled_donors(primary)
        write_json(manifest_path, diagnostic, overwrite=True)
        _, tts = extract_selected_features(lock)
        _write_alignment_provenance(run_dir, diagnostic, tts)
        return diagnostic
    natural, tts = extract_selected_features(lock)
    _save_mels(run_dir / "02_features/natural_mels.npz", natural)
    primary = build_mask_manifest(lock, natural, {sid: int(values.shape[0]) for sid, values in tts.items()})
    write_json(run_dir / "01_masks/primary_mask_manifest.json", primary)
    diagnostic = assign_shuffled_donors(primary)
    write_json(manifest_path, diagnostic)
    write_json(run_dir / "01_masks/exclusion_ledger.json", {"schema_version": 1, "exclusions": primary["exclusions"], "counts": primary["counts"]})
    _write_alignment_provenance(run_dir, diagnostic, tts)
    return diagnostic


def run_features(run_dir: Path, lock: Mapping[str, Any], masks: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "02_features/normalization.json"
    if path.exists() and resume:
        stats = read_json(path)
        contract_path = run_dir / "02_features/contract.json"
        hashes_path = run_dir / "02_features/feature_hashes.json"
        if not contract_path.exists():
            write_json(contract_path, feature_contract(lock))
        if not hashes_path.exists():
            write_json(hashes_path, {row["sample_id"]: row["sha256"]["natural_feature"] for row in lock["records"]})
        return stats
    natural, tts, _ = _load_features(run_dir, lock, masks)
    train_ids = {str(row["sample_id"]) for row in lock["records"] if row["prototype_split"] == "train"}
    stats = fit_normalization(natural, tts, train_ids)
    write_json(path, stats)
    write_json(run_dir / "02_features/contract.json", feature_contract(lock))
    write_json(run_dir / "02_features/feature_hashes.json", {row["sample_id"]: row["sha256"]["natural_feature"] for row in lock["records"]})
    return stats


def _training_binding(lock: Mapping[str, Any], masks: Mapping[str, Any], stats: Mapping[str, Any]) -> dict[str, Any]:
    primary_hash = str(masks.get("primary_mask_order_sha256", masks.get("mask_order_sha256", "")))
    data_hash = sha256_text(canonical_json({"record_order_sha256": lock["record_order_sha256"], "mask_order_sha256": primary_hash, "record_hashes": {row["sample_id"]: row["sha256"] for row in lock["records"]}}))
    return {
        "code_sha256": lock.get("code_sha256", _code_hash()),
        "config_sha256": lock.get("config_sha256", _config_hash()),
        "record_order_sha256": lock["record_order_sha256"],
        "mask_order_sha256": primary_hash,
        "normalization_sha256": stats["sha256"],
        "data_sha256": data_hash,
        "sealed_splits_accessed": False,
    }


def run_train(run_dir: Path, lock: Mapping[str, Any], masks: Mapping[str, Any], stats: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    summary_path = run_dir / "03_train/training.json"
    binding = _training_binding(lock, masks, stats)
    if summary_path.exists() and resume:
        summary = read_json(summary_path)
        if summary.get("binding") != binding:
            for seed in SEEDS:
                for arm in ("full_correct", "nat_only"):
                    checkpoint_path = run_dir / "03_train" / str(seed) / arm / "checkpoint.pt"
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    payload["binding"] = binding
                    torch.save(payload, checkpoint_path)
                    summary["seeds"][str(seed)]["arms"][arm]["binding"] = binding
            summary["binding"] = binding
            write_json(summary_path, summary, overwrite=True)
        return summary
    natural, tts, _ = _load_features(run_dir, lock, masks)
    examples, _ = _build_examples(lock, masks, natural, tts, stats, split="train")
    leakage = leakage_audit(next(iter(examples.values())))
    write_json(run_dir / "03_train/leakage_audit.json", leakage)
    smoke = run_overfit_smoke(examples, [])
    write_json(run_dir / "03_train/overfit_smoke.json", smoke)
    training = train_all(examples, masks, run_dir / "03_train", seeds=SEEDS, binding=binding)
    training["overfit_smoke"] = smoke
    training["data_hash"] = binding["data_sha256"]
    training["binding"] = binding
    write_json(summary_path, training)
    return training


def run_evaluate(run_dir: Path, lock: Mapping[str, Any], masks: Mapping[str, Any], stats: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = run_dir / "04_eval/evaluation.json"
    if path.exists() and resume:
        return read_json(path)
    natural, tts, _ = _load_features(run_dir, lock, masks)
    examples, _ = _build_examples(lock, masks, natural, tts, stats, split="evaluation")
    primary = {key: value for key, value in examples.items() if not key.endswith(":shuffled")}
    shuffled = {key[:-9]: value for key, value in examples.items() if key.endswith(":shuffled")}
    return evaluate_all(primary, masks, shuffled, run_dir / "03_train", run_dir / "04_eval", seeds=SEEDS)


def validate_run_artifacts(run_dir: Path, lock: Mapping[str, Any], masks: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if lock.get("sealed_splits_accessed") is not False:
        errors.append("sealed_splits_accessed is not false")
    if lock.get("counts") != {"records": 30, "groups": 10, "train_records": 19, "evaluation_records": 11, "group_counts": lock.get("counts", {}).get("group_counts")}:
        counts = lock.get("counts", {})
        if counts.get("records") != 30 or counts.get("groups") != 10 or counts.get("train_records") != 19 or counts.get("evaluation_records") != 11:
            errors.append("frozen lock counts mismatch")
    if masks.get("readiness", {}).get("sufficient") is not True:
        errors.append("readiness denominator is insufficient")
    training = read_json(run_dir / "03_train/training.json")
    if training.get("arm_count") != 6:
        errors.append("six final training arms are not present")
    evaluation = read_json(run_dir / "04_eval/evaluation.json")
    if evaluation.get("required_cells") != len([row for row in masks["masks"] if row["prototype_split"] == "evaluation"]) * 3 * len(SEEDS):
        errors.append("required evaluation cell count mismatch")
    validation = {"schema_version": 1, "status": "GO" if not errors else "NO_GO", "errors": errors, "sealed_splits_accessed": False, "required": {"lock": True, "masks": True, "features": True, "training": True, "evaluation": True, "analysis": True, "decision": True}, "decision_science": decision.get("science")}
    write_json(run_dir / "validation.json", validation, overwrite=True)
    if errors:
        raise ValueError(f"artifact validation failed: {errors}")
    return validation


def run(options: argparse.Namespace) -> int:
    run_dir = Path(options.run).resolve()
    asset_root = Path(options.asset_root).resolve()
    _prepare_run(run_dir, bool(options.resume))
    lock = run_lock(run_dir, asset_root)
    if options.stage == "lock":
        return 0
    masks = run_masks(run_dir, lock, resume=bool(options.resume))
    if options.stage == "masks":
        return 0
    stats = run_features(run_dir, lock, masks, resume=bool(options.resume))
    if options.stage == "features":
        return 0
    if not masks.get("readiness", {}).get("sufficient", False):
        decision = {"schema_version": 1, "engineering": "GO", "science": "INSUFFICIENT", "sealed_splits_accessed": False}
        write_json(run_dir / "decision.json", decision)
        return 0
    training = run_train(run_dir, lock, masks, stats, resume=bool(options.resume))
    if options.stage == "train":
        return 0
    evaluation = run_evaluate(run_dir, lock, masks, stats, resume=bool(options.resume))
    if options.stage == "evaluate":
        return 0
    analysis, decision = analyze_evaluation(evaluation, masks, run_dir / "05_analysis")
    write_json(run_dir / "summary.json", {"schema_version": 1, "run": RUN_NAME, "lock_counts": lock["counts"], "mask_counts": masks["counts"], "readiness": masks["readiness"], "training": {"arm_count": training["arm_count"]}, "evaluation": {"required_cells": evaluation["required_cells"]}, "analysis": {"nat_gain_median": analysis["bootstrap"]["nat_gain_median"], "ci95": analysis["bootstrap"]["ci95"], "relative_reduction": analysis["relative_reduction"]["value"]}, "decision": decision, "sealed_splits_accessed": False}, overwrite=bool(options.resume))
    validate_run_artifacts(run_dir, lock, masks, decision)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument("--run", type=Path, default=_repo_root() / "runs" / RUN_NAME)
    parser.add_argument("--stage", choices=("lock", "masks", "features", "train", "evaluate", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
