#!/usr/bin/env python3
"""Evaluate a frozen TTS feature adapter on an untouched split.

The evaluator intentionally reports representation-only metrics.  It does not
run Ditto/TFG/SyncNet and it never changes the checkpoint or fits statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tts_feature_head import ResidualFeatureAdapter
    from train_tts_feature_head import (
        DataContractError,
        PrototypeTable,
        _records_for_split,
        build_frame_batches,
        evaluate_batches,
        evaluate_identity,
        load_manifest,
        sha256_file,
        sha256_json,
        write_json,
    )
except ImportError:  # pragma: no cover
    from scripts.tts_feature_head import ResidualFeatureAdapter
    from scripts.train_tts_feature_head import (
        DataContractError,
        PrototypeTable,
        _records_for_split,
        build_frame_batches,
        evaluate_batches,
        evaluate_identity,
        load_manifest,
        sha256_file,
        sha256_json,
        write_json,
    )


def _table_from_checkpoint(value: dict[str, Any]) -> PrototypeTable:
    labels = [str(label) for label in value["labels"]]
    natural = {label: np.asarray(value["natural_centroid"][label], dtype=np.float32) for label in labels}
    tts = {label: np.asarray(value["tts_centroid"][label], dtype=np.float32) for label in labels}
    delta = {label: np.asarray(value["delta"][label], dtype=np.float32) for label in labels}
    dim = int(value["dim"])
    for mapping in (natural, tts, delta):
        if any(array.shape != (dim,) or not np.isfinite(array).all() for array in mapping.values()):
            raise DataContractError("checkpoint contains invalid prototype vectors")
    return PrototypeTable(
        natural=natural,
        tts=tts,
        delta=delta,
        support_natural={label: int(value["support_natural"][label]) for label in labels},
        support_tts={label: int(value["support_tts"][label]) for label in labels},
        dim=dim,
        min_support=int(value["min_support"]),
        alpha=float(value["alpha"]),
    )


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[ResidualFeatureAdapter, PrototypeTable, dict[str, Any]]:
    """Load model, train-only prototypes, and immutable checkpoint metadata."""
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DataContractError("unsupported or malformed feature-head checkpoint")
    model_config = dict(payload["model_config"])
    model = ResidualFeatureAdapter(
        input_dim=int(model_config["input_dim"]),
        hidden_channels=int(model_config["hidden_channels"]),
        dilations=tuple(int(value) for value in model_config["dilations"]),
        residual_scale=float(model_config["residual_scale"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    prototypes = _table_from_checkpoint(payload["prototype"])
    if prototypes.dim != model.input_dim:
        raise DataContractError("checkpoint prototype dimension and model input dimension differ")
    return model, prototypes, payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint: Path,
    manifest: Path,
    embedding_dir: Path,
    split: str,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Evaluate one frozen checkpoint with explicit held-out provenance."""
    device = torch.device(device_name)
    model, prototypes, payload = load_model_checkpoint(checkpoint, device)
    records, manifest_meta = load_manifest(manifest)
    expected_manifest_hash = payload.get("metadata", {}).get("manifest_sha256")
    actual_manifest_hash = manifest_meta["manifest_sha256"]
    if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
        raise DataContractError(
            f"manifest hash mismatch: checkpoint={expected_manifest_hash} supplied={actual_manifest_hash}"
        )
    split_records = _records_for_split(records, split)
    if not split_records:
        raise DataContractError(f"no records for split {split}")
    split_keys = sorted({_record_key(record) for record in split_records})
    checkpoint_keys = payload.get("metadata", {}).get("split_keys", {})
    expected_keys = sorted(checkpoint_keys.get(split, []))
    if expected_keys and split_keys != expected_keys:
        raise DataContractError("supplied split key list differs from checkpoint provenance")
    batches, diagnostics = build_frame_batches(
        split_records,
        embedding_dir,
        prototypes,
        layer=int(payload["run_config"]["layer"]),
        input_condition="natural",
    )
    identity = evaluate_identity(batches, prototypes, device)
    adapter = evaluate_batches(model, batches, prototypes, device)
    result = {
        "schema_version": 1,
        "representation_only": True,
        "downstream_evaluation": "not_run",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(manifest),
        "manifest_sha256": actual_manifest_hash,
        "embedding_dir": str(embedding_dir),
        "embedding_dir_note": "individual embedding hashes are retained in source sidecars; directory hash is not used",
        "split": split,
        "held_out_keys": split_keys,
        "held_out_key_count": len(split_keys),
        "leakage_checks": {
            "checkpoint_manifest_match": True,
            "split_key_match": bool(not expected_keys or split_keys == expected_keys),
            "prototype_hash": sha256_json(prototypes.to_json()),
            "prototype_source": "checkpoint train-only prototypes",
            "test_used_for_model_selection": False,
        },
        "diagnostics": diagnostics,
        "identity_baseline": identity,
        "adapter": adapter,
        "warnings": payload.get("warnings", []),
    }
    return result


def _record_key(record: dict[str, Any]) -> str:
    value = record.get("paired_key") or record.get("pair_key") or record.get("utterance_id")
    if value is None:
        raise DataContractError("record has no paired key")
    return str(value)


def main() -> None:
    args = _parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        args.embedding_dir,
        args.split,
        args.device,
    )
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "split": result["split"], "held_out_key_count": result["held_out_key_count"], "adapter_loss": result["adapter"]["adapter_loss"], "identity_loss": result["identity_baseline"]["loss"]}, sort_keys=True))


if __name__ == "__main__":
    main()
