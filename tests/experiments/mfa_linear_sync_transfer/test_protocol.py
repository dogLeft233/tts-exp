from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.experiments.mfa_linear_sync_transfer.config import (
    COHORT_SIZE,
    GAIN_MIN,
    SELECTION_SALT,
    TransferConfig,
    validate_serialized_config,
)
from scripts.experiments.mfa_linear_sync_transfer.protocol import (
    FrozenTTSOnlyAdapter,
    TransferProtocolError,
    select_cohort,
    selection_key,
    validate_p2_parent,
)
from scripts.experiments.mfa_linear_real_video_sync.model import build_waveform_model, module_state_sha256
from scripts.experiments.mfa_linear_sync_transfer.validate import validate_cohort_manifest


def _row(group: str, index: int) -> dict[str, object]:
    sid = f"lrs3_{group}_{index:05d}"
    return {
        "sample_id": sid,
        "source_group": group,
        "protocol_split": "train",
        "selection_key": selection_key(group, sid),
        "visual": {"path": f"{sid}.avi", "sha256": "x", "decoded_bgr_frame_sha256": ["x"] * 96},
        "natural_target": {"target_offset": -1, "best_second_gap": 0.01, "curve": [1.0] * 31},
        "mfa_audio": f"{sid}.wav",
    }


def test_frozen_serialized_config_rejects_tuning_fields() -> None:
    payload = TransferConfig().to_dict()
    validate_serialized_config(payload)
    payload["gain_min"] = 0.0
    with pytest.raises(ValueError, match="frozen"):
        validate_serialized_config(payload)
    payload = TransferConfig().to_dict()
    payload["extra_seed"] = 123
    with pytest.raises(ValueError, match="frozen"):
        validate_serialized_config(payload)


def test_selection_is_score_independent_and_one_per_group(tmp_path: Path) -> None:
    rows = [_row("group-a", 1), _row("group-a", 2)] + [_row(f"group-{i}", 1) for i in range(7)]
    manifest = select_cohort(rows, [], tmp_path)
    assert manifest["selection_salt"] == SELECTION_SALT
    assert manifest["denominator"] == COHORT_SIZE
    assert len(manifest["selected"]) == COHORT_SIZE
    assert len(set(manifest["selected_source_groups"])) == COHORT_SIZE
    assert manifest["selection_uses_outcomes"] is False
    assert validate_cohort_manifest(tmp_path / "01_cohort_lock/manifest.json")["status"] == "valid"
    expected = min((row for row in rows if row["source_group"] == "group-a"), key=lambda row: row["selection_key"])
    selected = next(row for row in manifest["selected"] if row["source_group"] == "group-a")
    assert selected["sample_id"] == expected["sample_id"]


def test_selection_rejects_insufficient_groups(tmp_path: Path) -> None:
    with pytest.raises(TransferProtocolError, match="BLOCKED_COHORT_LOCK"):
        select_cohort([_row("only", 1)], [], tmp_path)
    assert not (tmp_path / "01_cohort_lock/manifest.json").exists()


def test_frozen_adapter_accepts_only_exact_waveform() -> None:
    model = build_waveform_model(seed=20_260_903, device="cpu")
    adapter = FrozenTTSOnlyAdapter(model)
    before = module_state_sha256(adapter)
    output = adapter(torch.zeros(1, 1, 61_440))
    assert output.shape == (1, 1, 61_440)
    assert module_state_sha256(adapter) == before
    with pytest.raises(TransferProtocolError, match="accepts only"):
        adapter(torch.zeros(1, 2, 61_440))
    with pytest.raises(TransferProtocolError, match="accepts only"):
        adapter({"mfa_linear_tts_waveform": torch.zeros(1, 1, 61_440)})


def test_parent_loader_rejects_sealed_p1() -> None:
    root = Path(__file__).resolve().parents[3] / "runs/lrs3_mfa_linear_real_video_sync_20260903_v4"
    with pytest.raises(TransferProtocolError, match="BLOCKED_P2_NOT_PASSED"):
        validate_p2_parent(root, repo=root.parents[1])


def test_parent_loader_rejects_missing_parent(tmp_path: Path) -> None:
    with pytest.raises(TransferProtocolError, match="BLOCKED_P2_NOT_PASSED"):
        validate_p2_parent(tmp_path)
