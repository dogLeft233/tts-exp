"""Held-out paired conditions using unchanged final checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .config import SEEDS, WAVLM_DIM
from .model import MaskedNaturalReconstructor, reconstruction_loss
from .protocol import canonical_json, sha256_file, sha256_text, write_json
from .train import load_checkpoint


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _to_batch(example: Mapping[str, np.ndarray], *, zero_tts: bool) -> dict[str, torch.Tensor]:
    batch = {}
    for key in ("natural_mel", "masked_support", "target_core", "target"):
        value = np.asarray(example[key], dtype=np.float32)
        batch[key] = torch.from_numpy(value[None, ...])
    tts = np.asarray(example["tts_features"], dtype=np.float32).copy()
    if zero_tts:
        tts[...] = 0.0
    batch["tts_features"] = torch.from_numpy(tts[None, ...])
    return batch


def score_condition(model: MaskedNaturalReconstructor, example: Mapping[str, np.ndarray], *, zero_tts: bool) -> tuple[np.ndarray, dict[str, float]]:
    batch = _to_batch(example, zero_tts=zero_tts)
    with torch.no_grad():
        prediction = model(batch["natural_mel"], batch["masked_support"], batch["target_core"], batch["tts_features"])
        losses = reconstruction_loss(prediction, batch["target"], batch["target_core"])
    prediction_array = prediction[0].cpu().numpy().astype(np.float32, copy=False)
    loss_values = {key: float(value.cpu()) for key, value in losses.items()}
    if not np.isfinite(prediction_array).all() or not all(np.isfinite(value) for value in loss_values.values()):
        raise FloatingPointError("non-finite held-out prediction or loss")
    return prediction_array, loss_values


def evaluate_all(
    examples: Mapping[str, Mapping[str, np.ndarray]],
    masks: Mapping[str, Any],
    shuffled_examples: Mapping[str, Mapping[str, np.ndarray]],
    train_dir: Path,
    output_dir: Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_masks = [row for row in masks["masks"] if row["prototype_split"] == "evaluation"]
    evaluation_masks.sort(key=lambda row: int(row["canonical_index"]))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        full, full_payload = load_checkpoint(train_dir / str(seed) / "full_correct" / "checkpoint.pt")
        natural_only, natural_payload = load_checkpoint(train_dir / str(seed) / "nat_only" / "checkpoint.pt")
        if full_payload["initial_state_sha256"] != natural_payload["initial_state_sha256"] or full_payload["schedule_sha256"] != natural_payload["schedule_sha256"]:
            raise ValueError(f"initial state or schedule mismatch for seed {seed}")
        for mask in evaluation_masks:
            mask_hash = str(mask["mask_sha256"])
            if mask_hash not in examples:
                raise ValueError(f"missing primary evaluation example {mask_hash}")
            conditions: list[tuple[str, MaskedNaturalReconstructor, Mapping[str, np.ndarray], bool]] = [
                ("FULL_CORRECT", full, examples[mask_hash], False),
                ("FULL_ZERO", full, examples[mask_hash], True),
                ("NAT_ONLY", natural_only, examples[mask_hash], True),
            ]
            shuffled = masks_by_hash(masks).get(mask_hash, {}).get("shuffled_donor", {})
            if shuffled.get("status") == "assigned" and mask_hash in shuffled_examples:
                conditions.append(("FULL_SHUFFLED", full, shuffled_examples[mask_hash], False))
            elif shuffled.get("status") == "diagnostic_missing":
                rows.append({"schema_version": 1, "seed": int(seed), "mask_sha256": mask_hash, "condition": "FULL_SHUFFLED", "status": "diagnostic_missing"})
            else:
                raise ValueError(f"shuffled diagnostic assignment is inconsistent for {mask_hash}")
            for condition, model, example, zero_tts in conditions:
                prediction, loss = score_condition(model, example, zero_tts=zero_tts)
                cell_dir = output_dir / str(seed) / mask_hash / condition
                cell_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cell_dir / "prediction.npz", prediction=prediction)
                checkpoint_path = train_dir / str(seed) / ("nat_only" if condition == "NAT_ONLY" else "full_correct") / "checkpoint.pt"
                record = {
                    "schema_version": 1,
                    "seed": int(seed),
                    "sample_id": str(mask["sample_id"]),
                    "source_group": str(mask["source_group"]),
                    "mask_sha256": mask_hash,
                    "condition": condition,
                    "status": "complete",
                    "loss": loss,
                    "prediction_shape": list(prediction.shape),
                    "prediction_sha256": array_hash(prediction),
                    "target_sha256": array_hash(np.asarray(example["target"], dtype=np.float32)),
                    "natural_input_sha256": array_hash(np.asarray(example["natural_mel"], dtype=np.float32)),
                    "masked_support_sha256": array_hash(np.asarray(example["masked_support"], dtype=np.float32)),
                    "target_core_sha256": array_hash(np.asarray(example["target_core"], dtype=np.float32)),
                    "tts_input_sha256": array_hash(np.zeros_like(example["tts_features"]) if zero_tts else np.asarray(example["tts_features"], dtype=np.float32)),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
                write_json(cell_dir / "record.json", record)
                rows.append(record)
    required = [row for row in rows if row.get("condition") in {"FULL_CORRECT", "FULL_ZERO", "NAT_ONLY"}]
    expected = len(evaluation_masks) * len(seeds) * 3
    if len(required) != expected:
        raise ValueError(f"required evaluation cells expected {expected}, found {len(required)}")
    result = {"schema_version": 1, "status": "GO", "evaluation_masks": len(evaluation_masks), "seeds": list(seeds), "required_cells": expected, "records": rows, "diagnostic_missing": sum(row.get("status") == "diagnostic_missing" for row in rows), "sealed_splits_accessed": False}
    write_json(output_dir / "evaluation.json", result)
    return result


def masks_by_hash(mask_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["mask_sha256"]): row for row in mask_manifest["masks"]}
