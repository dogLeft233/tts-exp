from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_knn_vc_poc import (  # noqa: E402
    DEFAULT_ALIGNMENT_MANIFEST,
    DEFAULT_STAGE_MANIFEST,
    exact_natural_length,
    select_records,
)


def test_exact_natural_length_crops_and_pads_right() -> None:
    cropped, crop_meta = exact_natural_length(np.arange(5, dtype=np.float32), 3)
    padded, pad_meta = exact_natural_length(np.arange(2, dtype=np.float32), 4)
    np.testing.assert_array_equal(cropped, [0, 1, 2])
    np.testing.assert_array_equal(padded, [0, 1, 0, 0])
    assert crop_meta["action"] == "right_crop"
    assert pad_meta["action"] == "right_zero_pad"


@pytest.mark.parametrize("count", [1, 10, 15])
def test_real_valid_selector_is_stable_and_excludes_heldout(tmp_path: Path, count: int) -> None:
    selected, metadata, localized = select_records(
        DEFAULT_STAGE_MANIFEST, DEFAULT_ALIGNMENT_MANIFEST, tmp_path, count,
    )
    assert len(selected) == count
    assert metadata["speaker_group"] == "S0765"
    assert metadata["available_count"] == 50
    assert all(pair["natural"]["split"] == "valid" for pair in selected)
    assert all(pair["natural"]["speaker_id"] != "S0770" for pair in selected)
    assert localized.is_file()


def test_selector_rejects_unsupported_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 1, 10, or 15"):
        select_records(DEFAULT_STAGE_MANIFEST, DEFAULT_ALIGNMENT_MANIFEST, tmp_path, 2)
