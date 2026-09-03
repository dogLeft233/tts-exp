from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.asr_targeted_local_replacement.evaluate import (
    _local_diagnostic,
    validate_isolated_work_paths,
)


def test_shared_wav2lip_paths_fail_before_launch(tmp_path: Path):
    work = tmp_path / "cell"
    with pytest.raises(ValueError, match="shared"):
        validate_isolated_work_paths([
            {"work_dir": work, "temp_dir": work / "temp"},
            {"work_dir": work, "temp_dir": work / "temp"},
        ])


def test_fixed_coordinate_local_diagnostic_uses_natural_column():
    natural = np.full((20, 31), 3.0, dtype=np.float32)
    candidate = natural.copy()
    natural[:, 15] = 1.0
    candidate[8:12, 15] = 0.5
    metadata = {
        "selected_track_index": 0,
        "selected_frame_count": 80,
        "track_start_frame": 0,
        "track_end_frame": 79,
        "track_ranges": [{"track_index": 0, "frame_count": 80, "start_frame": 0, "end_frame": 79, "eligible": True}],
    }
    result = _local_diagnostic(metadata, metadata, natural, candidate, [{"destination_start_s": 0.3, "destination_end_s": 0.45}], audio_duration_s=4.0)
    assert result is not None
    assert result["status"] == "ok"
    assert result["coordinate"]["offset_column"] == 15


def test_track_mismatch_nulls_local_diagnostic():
    a = {"selected_track_index": 0}
    b = {"selected_track_index": 1}
    matrix = np.ones((20, 31), dtype=np.float32)
    assert _local_diagnostic(a, b, matrix, matrix, [], audio_duration_s=1.0) == {"status": "null", "reason": "paired_track_mismatch"}
