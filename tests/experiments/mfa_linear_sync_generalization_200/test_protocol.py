from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.experiments.mfa_linear_real_video_sync.model import build_waveform_model, module_state_sha256
from scripts.experiments.mfa_linear_sync_generalization_200.config import (
    EVAL_RECORD_COUNT,
    EXPECTED_MATRIX_CELLS,
    MIN_SUCCESS_RECORDS,
    TRAIN_RECORD_COUNT,
    ScaleConfig,
    validate_serialized_config,
)
from scripts.experiments.mfa_linear_sync_generalization_200.protocol import (
    FrozenTTSOnlyAdapter,
    ScaleProtocolError,
    select_scale_cohorts,
    selection_key,
)
from scripts.experiments.mfa_linear_sync_generalization_200.validate import validate_data_manifest


def _row(group: str, index: int) -> dict[str, object]:
    sample_id = f"lrs3_{group}_{index:05d}"
    return {
        "sample_id": sample_id,
        "source_group": group,
        "protocol_split": "train",
        "selection_key": "old-key",
        "mfa_audio": f"{sample_id}.wav",
        "mfa_audio_sha256": "mfa",
        "natural_target": {"target_offset": 0, "best_second_gap": 0.01, "curve": [1.0] * 31},
        "visual": {"path": f"{sample_id}.avi", "sha256": "video", "decoded_bgr_frame_sha256": ["x"] * 96},
    }


def test_scale_config_is_frozen() -> None:
    config = ScaleConfig().to_dict()
    assert config["train_record_count"] == TRAIN_RECORD_COUNT
    assert config["evaluation_record_count"] == EVAL_RECORD_COUNT
    assert config["minimum_success_records"] == MIN_SUCCESS_RECORDS
    assert config["expected_matrix_cells"] == EXPECTED_MATRIX_CELLS
    validate_serialized_config(config)
    config["evaluation_record_count"] = 8
    with pytest.raises(ValueError, match="frozen"):
        validate_serialized_config(config)


def test_selection_locks_200_training_and_40_evaluation_rows(tmp_path: Path) -> None:
    rows = [
        _row(f"group-{group:02d}", index)
        for group in range(60)
        for index in range(10)
    ]
    manifest = select_scale_cohorts(rows, [], tmp_path, excluded_source_groups=[])
    assert len(manifest["training"]) == TRAIN_RECORD_COUNT
    assert len(manifest["evaluation"]) == EVAL_RECORD_COUNT
    assert len(manifest["evaluation_groups"]) == EVAL_RECORD_COUNT
    assert not (
        {row["source_group"] for row in manifest["training"]}
        & set(manifest["evaluation_groups"])
    )
    assert all(
        row["selection_key"] == selection_key(
            str(row["source_group"]), str(row["sample_id"]), role="evaluation"
        )
        for row in manifest["evaluation"]
    )
    assert validate_data_manifest(tmp_path / "01_data_lock/manifest.json")["status"] == "valid"


def test_selection_blocks_without_40_groups(tmp_path: Path) -> None:
    rows = [_row(f"group-{group:02d}", 0) for group in range(39)]
    with pytest.raises(ScaleProtocolError, match="BLOCKED_DATASET_SCALE"):
        select_scale_cohorts(rows, [], tmp_path, excluded_source_groups=[])
    assert not (tmp_path / "01_data_lock/manifest.json").exists()


def test_selection_excludes_predecessor_groups(tmp_path: Path) -> None:
    rows = [
        _row("6ORDQFh0Byw", index) for index in range(20)
    ] + [
        _row(f"group-{group:02d}", index)
        for group in range(60)
        for index in range(10)
    ]
    manifest = select_scale_cohorts(rows, [], tmp_path)
    assert "6ORDQFh0Byw" not in manifest["training_source_groups"]
    assert "6ORDQFh0Byw" not in manifest["evaluation_groups"]


def test_adapter_is_exact_waveform_only() -> None:
    model = build_waveform_model(seed=20_260_903, device="cpu")
    adapter = FrozenTTSOnlyAdapter(model)
    before = module_state_sha256(adapter)
    output = adapter(torch.zeros(1, 1, 61_440))
    assert output.shape == (1, 1, 61_440)
    assert module_state_sha256(adapter) == before
    with pytest.raises(ScaleProtocolError, match="only"):
        adapter(torch.zeros(1, 2, 61_440))
    with pytest.raises(ScaleProtocolError, match="only"):
        adapter({"natural_audio": torch.zeros(1, 1, 61_440)})
