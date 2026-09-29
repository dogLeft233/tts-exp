from pathlib import Path

import pytest

from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation import config
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.common import (
    assert_run_root_compatible,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.protocol import (
    ProtocolError,
    load_frozen_cohort,
)


def test_registered_cohort_hash_and_matrix_shape() -> None:
    assert config.EXPECTED_RECORD_COUNT == 22
    assert config.EXPECTED_SOURCE_GROUP_COUNT == 22
    assert config.ARMS == ("N", "N_REPEAT", "LOCAL_SWAP", "BRIDGE_075")
    assert config.MATRIX_CELLS == (
        "V_N/A_N",
        "V_N_REPEAT/A_N",
        "V_LOCAL_SWAP/A_N",
        "V_LOCAL_SWAP/A_LOCAL_SWAP",
        "V_BRIDGE_075/A_N",
        "V_BRIDGE_075/A_BRIDGE_075",
    )
    assert config.EXPECTED_VIDEO_COUNT == 88
    assert config.EXPECTED_CELL_COUNT == 132


def test_registered_cohort_selects_second_records_and_is_disjoint() -> None:
    records, selection = load_frozen_cohort()
    assert len(records) == config.EXPECTED_RECORD_COUNT
    assert len({row["source_group"] for row in records}) == config.EXPECTED_SOURCE_GROUP_COUNT
    assert selection["selection"] == "second ordered record per source group from parent protocol cohort"
    assert selection["discovery_disjoint"] is True
    assert selection["sample_ids_sha256"] == config.EXPECTED_SAMPLE_ID_SHA256


def test_incompatible_populated_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "00_protocol").mkdir(parents=True)
    (root / "00_protocol" / "protocol.json").write_text('{"protocol_id":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="another protocol"):
        assert_run_root_compatible(root, config.PROTOCOL_ID)


def test_missing_parent_hash_is_rejected(tmp_path: Path) -> None:
    parent = tmp_path / "protocol.json"
    parent.write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="parent artifact hash changed"):
        load_frozen_cohort(parent, config.PARENT_REPLACEMENT)
