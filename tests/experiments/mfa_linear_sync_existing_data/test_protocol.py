from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.experiments.mfa_linear_real_video_sync.model import build_waveform_model, module_state_sha256
from scripts.experiments.mfa_linear_sync_existing_data.config import (
    EVAL_RECORD_COUNT,
    EXPECTED_MATRIX_CELLS,
    MIN_SUCCESS_RECORDS,
    TRAIN_RECORD_COUNT,
    ExistingDataConfig,
    validate_serialized_config,
)
from scripts.experiments.mfa_linear_sync_existing_data.protocol import (
    ExistingDataProtocolError,
    FrozenTTSOnlyAdapter,
    select_cohort,
    selection_key,
)
from scripts.experiments.mfa_linear_sync_existing_data.validate import validate_manifest


def _row(group: str, index: int) -> dict[str, object]:
    sample_id = f"lrs3_{group}_{index:05d}"
    return {
        "sample_id": sample_id,
        "source_group": group,
        "protocol_split": "train",
        "mfa_audio": f"{sample_id}.wav",
        "mfa_audio_sha256": "mfa",
        "natural_target": {"target_offset": 0, "best_second_gap": 0.01, "curve": [1.0] * 31},
        "visual": {"path": f"{sample_id}.avi", "sha256": "video", "decoded_bgr_frame_sha256": ["x"] * 96},
    }


def test_config_is_frozen() -> None:
    payload = ExistingDataConfig().to_dict()
    assert payload["train_record_count"] == TRAIN_RECORD_COUNT
    assert payload["evaluation_record_count"] == EVAL_RECORD_COUNT
    assert payload["minimum_success_records"] == MIN_SUCCESS_RECORDS
    assert payload["expected_matrix_cells"] == EXPECTED_MATRIX_CELLS
    validate_serialized_config(payload)
    payload["train_record_count"] = 200
    with pytest.raises(ValueError, match="frozen"):
        validate_serialized_config(payload)


def test_record_heldout_selection_keeps_source_group_overlap_explicit(tmp_path: Path) -> None:
    rows = [_row("group-00", index) for index in range(12)]
    rows += [_row(f"group-{group:02d}", index) for group in range(1, 10) for index in range(10)]
    manifest = select_cohort(rows, [], tmp_path, excluded_source_groups=[])
    assert len(manifest["training"]) == TRAIN_RECORD_COUNT
    assert len(manifest["evaluation"]) == EVAL_RECORD_COUNT
    assert len(set(row["sample_id"] for row in manifest["training"]) & set(row["sample_id"] for row in manifest["evaluation"])) == 0
    assert manifest["source_group_overlap"]
    assert all(
        row["selection_key"] == selection_key(str(row["source_group"]), str(row["sample_id"]), role="evaluation")
        for row in manifest["evaluation"]
    )
    assert validate_manifest(tmp_path / "01_data_lock/manifest.json")["status"] == "valid"


def test_selection_requires_exact_existing_inventory(tmp_path: Path) -> None:
    rows = [_row(f"group-{group:02d}", 0) for group in range(8)]
    with pytest.raises(ExistingDataProtocolError, match="BLOCKED_EXISTING_DATA"):
        select_cohort(rows, [], tmp_path, excluded_source_groups=[])
    assert not (tmp_path / "01_data_lock/manifest.json").exists()


def test_adapter_accepts_only_exact_waveform() -> None:
    model = build_waveform_model(seed=20_260_903, device="cpu")
    adapter = FrozenTTSOnlyAdapter(model)
    before = module_state_sha256(adapter)
    output = adapter(torch.zeros(1, 1, 61_440))
    assert output.shape == (1, 1, 61_440)
    assert module_state_sha256(adapter) == before
    with pytest.raises(ExistingDataProtocolError, match="only"):
        adapter(torch.zeros(1, 2, 61_440))
    with pytest.raises(ExistingDataProtocolError, match="only"):
        adapter({"natural_audio": torch.zeros(1, 1, 61_440)})
