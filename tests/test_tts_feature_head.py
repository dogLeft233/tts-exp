"""Tests for the train-only TTS feature-domain enhancement MVP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_tts_feature_head import _table_from_checkpoint
from tts_feature_head import ResidualFeatureAdapter, feature_statistics, parameter_statistics
from train_tts_feature_head import (
    DataContractError,
    FrameBatch,
    PrototypeTable,
    RunConfig,
    build_phone_prototypes,
    compute_feature_loss,
    evaluate_identity,
    set_seed,
    validate_pair_splits,
)


def _record(key: str, condition: str, split: str, speaker: str, embedding_path: str, *, tokens=None) -> dict:
    return {
        "paired_key": key,
        "sample_id": key,
        "condition": condition,
        "split": split,
        "speaker_id": speaker,
        "alignment_source": "mfa",
        "alignment_method": "mandarin_mfa",
        "embedding_path": embedding_path,
        "tokens": tokens or [
            {"token": "a", "start_s": 0.0, "end_s": 0.09},
            {"token": "i", "start_s": 0.09, "end_s": 0.19},
        ],
    }


def _write_embedding(tmp_path: Path, record: dict, values: np.ndarray, layer: int = 11) -> None:
    path = tmp_path / f"{record['condition']}_{record['paired_key']}.npy"
    np.save(path, values.astype(np.float32))
    sidecar = {
        "layers": [layer],
        "embedding_dim": values.shape[-1],
        "frame_times_s": [0.04, 0.14],
        "sample_rate": 16000,
        "frame_stride_samples": 320,
        "allow_implicit_frame_times": False,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    record["embedding_path"] = str(path)


def _table(dim: int = 3) -> PrototypeTable:
    natural = {"a": np.zeros(dim, dtype=np.float32), "i": np.ones(dim, dtype=np.float32)}
    tts = {"a": np.full(dim, 2, dtype=np.float32), "i": np.full(dim, 3, dtype=np.float32)}
    delta = {label: tts[label] - natural[label] for label in natural}
    return PrototypeTable(natural, tts, delta, {"a": 10, "i": 10}, {"a": 10, "i": 10}, dim, 1, 0.5)


def test_fresh_adapter_is_exact_identity_and_bounded() -> None:
    torch.manual_seed(0)
    model = ResidualFeatureAdapter(input_dim=4, hidden_channels=8, dilations=(1, 2), residual_scale=0.2)
    x = torch.randn(2, 7, 4)
    with torch.no_grad():
        y = model(x)
    assert torch.equal(x, y)
    assert y.shape == x.shape
    assert torch.max(torch.abs(y - x)) == 0


def test_adapter_mask_keeps_invalid_frames_unchanged() -> None:
    model = ResidualFeatureAdapter(input_dim=3, hidden_channels=4, dilations=(1,))
    x = torch.randn(1, 4, 3)
    mask = torch.tensor([[True, False, True, False]])
    with torch.no_grad():
        y = model(x, mask)
    assert torch.equal(y[:, 1], x[:, 1])
    assert torch.equal(y[:, 3], x[:, 3])


def test_adapter_rejects_bad_shape_and_nonfinite() -> None:
    model = ResidualFeatureAdapter(input_dim=3, hidden_channels=4, dilations=(1,))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="finite"):
        model(torch.tensor([[[float("nan"), 0.0, 0.0]]]))


def test_feature_statistics_and_grad_debug_are_finite() -> None:
    model = ResidualFeatureAdapter(input_dim=3, hidden_channels=4, dilations=(1,))
    x = torch.randn(1, 4, 3, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    stats = feature_statistics(y)
    debug = parameter_statistics(model)
    assert stats["finite_count"] == 12
    assert np.isfinite(list(debug.values())).all()


def test_loss_uses_alpha_shift_and_identity_is_not_shifted() -> None:
    table = _table()
    source = torch.zeros(1, 2, 3)
    prediction = torch.tensor([[[1.0, 1.0, 1.0], [1.5, 1.5, 1.5]]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    total, terms, debug = compute_feature_loss(prediction, source, ["a", "i"], mask, table)
    assert torch.isfinite(total)
    assert terms["prototype_shift_smooth_l1"] >= 0
    assert debug["valid_frames"] == 2
    identity = evaluate_identity(
        [FrameBatch("k", "s", "valid", source.detach().numpy()[0], np.array(["a", "i"]), np.array([True, True]), np.array([0.04, 0.14]), {})],
        table,
        torch.device("cpu"),
    )
    assert identity["identity_baseline"] is True
    assert identity["loss"] > 0
    total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_validate_pair_splits_rejects_leakage_and_uniform_fallback() -> None:
    records = [
        _record("p1", "natural", "train", "s1", "n"),
        _record("p1", "tts", "train", "s1", "t"),
        _record("p2", "natural", "test", "s2", "n"),
        _record("p2", "tts", "test", "s2", "t"),
    ]
    result = validate_pair_splits(records)
    assert result["split_pair_counts"] == {"train": 1, "valid": 0, "test": 1}
    records[3]["split"] = "train"
    with pytest.raises(DataContractError, match="split mismatch"):
        validate_pair_splits(records)
    records[3]["split"] = "test"
    records[2]["alignment_method"] = "uniform_fallback"
    with pytest.raises(DataContractError, match="uniform"):
        validate_pair_splits(records)


def test_validate_pair_splits_rejects_speaker_overlap() -> None:
    records = [
        _record("p1", "natural", "train", "same", "n"),
        _record("p1", "tts", "train", "same", "t"),
        _record("p2", "natural", "test", "same", "n"),
        _record("p2", "tts", "test", "same", "t"),
    ]
    with pytest.raises(DataContractError, match="speaker"):
        validate_pair_splits(records)


def test_prototypes_are_train_only_and_two_sided_supported(tmp_path: Path) -> None:
    records = []
    for condition in ("natural", "tts"):
        record = _record("p1", condition, "train", "s1", "")
        values = np.array([[0, 0], [0, 0]], dtype=np.float32) if condition == "natural" else np.array([[2, 2], [2, 2]], dtype=np.float32)
        _write_embedding(tmp_path, record, values, layer=11)
        records.append(record)
    table, diagnostics = build_phone_prototypes(records, tmp_path, layer=11, min_support=1, alpha=0.5)
    assert table.dim == 2
    assert set(table.labels) == {"a", "i"}
    assert np.allclose(table.delta["a"], [2, 2])
    assert diagnostics["support"]["alpha"] == 0.5
    non_train = [dict(record, split="test") for record in records]
    with pytest.raises(DataContractError, match="non-train"):
        build_phone_prototypes(non_train, tmp_path, layer=11, min_support=1)


def test_prototype_builder_rejects_layer_mismatch(tmp_path: Path) -> None:
    records = []
    for condition in ("natural", "tts"):
        record = _record("p1", condition, "train", "s1", "")
        _write_embedding(tmp_path, record, np.ones((2, 2), dtype=np.float32), layer=6)
        records.append(record)
    with pytest.raises(DataContractError, match="requested layer"):
        build_phone_prototypes(records, tmp_path, layer=11, min_support=1)


def test_checkpoint_prototype_round_trip() -> None:
    table = _table()
    restored = _table_from_checkpoint(table.to_json())
    assert restored.labels == table.labels
    assert restored.dim == table.dim
    for label in table.labels:
        assert np.array_equal(restored.delta[label], table.delta[label])


def test_seed_is_reproducible() -> None:
    set_seed(7)
    first = torch.randn(5)
    set_seed(7)
    second = torch.randn(5)
    assert torch.equal(first, second)
