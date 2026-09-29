from pathlib import Path

import pytest

from scripts.experiments.lrs3_local_timing_control_calibration.common import (
    CalibrationError,
)
from scripts.experiments.lrs3_local_timing_control_calibration.scoring import (
    parse_syncnet,
)


def test_syncnet_parser_requires_exact_metrics(tmp_path: Path) -> None:
    path = tmp_path / "score.log"
    path.write_text("Confidence: 1.2\nMin dist: 3.4\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="missing or ambiguous"):
        parse_syncnet(path)


def test_syncnet_parser_rejects_ambiguous_logs(tmp_path: Path) -> None:
    path = tmp_path / "score.log"
    path.write_text("Confidence: 1.2\nConfidence: 1.3\nMin dist: 3.4\nAV offset: 0\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="missing or ambiguous"):
        parse_syncnet(path)
