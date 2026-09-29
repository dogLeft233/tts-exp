from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiments.lrs3_bridge_tts_quality.common import ProtocolError
from scripts.experiments.lrs3_bridge_tts_quality.quality import (
    _paired_quality_rows,
    _read_ratings,
)


def _mapping() -> list[dict]:
    return [
        {"stimulus_id": "raw_local", "sample_id": "one", "stage": "raw", "provider": "LOCAL"},
        {"stimulus_id": "raw_cloud", "sample_id": "one", "stage": "raw", "provider": "CLOUD"},
    ]


def test_quality_pair_requires_common_raters() -> None:
    paired, missing = _paired_quality_rows(
        [{"rater_id": "r1", "stimulus_id": "raw_local", "quality": 4.0}],
        _mapping(),
        "raw",
    )
    assert paired == []
    assert missing[0]["common_rater_count"] == 0


def test_ratings_reject_out_of_range_and_provider_labels(tmp_path: Path) -> None:
    invalid_range = tmp_path / "range.csv"
    invalid_range.write_text("rater_id,stimulus_id,quality\nr1,s,6\n", encoding="utf-8")
    with pytest.raises(ProtocolError):
        _read_ratings(invalid_range, {"s"})
    labeled = tmp_path / "labeled.csv"
    labeled.write_text("rater_id,stimulus_id,quality,provider\nr1,s,4,CLOUD\n", encoding="utf-8")
    with pytest.raises(ProtocolError):
        _read_ratings(labeled, {"s"})
