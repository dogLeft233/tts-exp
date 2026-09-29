from pathlib import Path

import pytest

from scripts.experiments.lrs3_phase_preserving_replacement_envelope import config
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.common import (
    assert_run_root_compatible,
)
from scripts.experiments.lrs3_phase_preserving_replacement_envelope.protocol import (
    ProtocolError,
    load_frozen_cohort,
)


def test_registered_cohort_hash_and_matrix_shape() -> None:
    assert config.EXPECTED_RECORD_COUNT == 23
    assert config.EXPECTED_SOURCE_GROUP_COUNT == 23
    assert len(config.ARMS) == 7
    assert len(config.MATRIX_CELLS) == 13
    assert config.EXPECTED_CELL_COUNT == 299


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
