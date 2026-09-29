from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.experiments.tts_evidence_temporal_patch.evaluation import (
    export_blind_pack,
    import_official,
    score_temporal_rank,
)


def test_rank_uses_disjoint_calibration_and_evaluation_support() -> None:
    matrix = np.ones((100, 31), dtype=np.float32) * 5
    matrix[:, 15] = 0
    result = score_temporal_rank(matrix, guard_rows=20, min_calibration_rows=10, min_evaluation_rows=25)
    assert result["status"] == "MEASURABLE"
    assert result["lag"] == 0
    assert max(result["calibration_rows"]) < min(result["evaluation_rows"])
    assert result["evaluation_rows"][0] >= len(matrix) // 3 + 20


def test_official_import_preserves_fulltrack_name(tmp_path: Path) -> None:
    summary_path = tmp_path / "07_official_syncnet/full/summary.json"
    summary_path.parent.mkdir(parents=True)
    records = []
    for index in range(108):
        group = f"g{index % 36:02d}"
        arm = "natural" if index % 3 == 0 else ("qwen_cloud" if index % 3 == 1 else "qwen_local")
        records.append({"cell_key": f"{group}::s{index:03d}::{arm}", "source_group": group, "sample_id": f"s{index:03d}", "arm": arm, "sync_c": 7.0, "sync_d": 5.0, "av_offset": -2, "min_track": 25, "scorer": "official_syncnet_v2"})
    summary_path.write_text(json.dumps({"completed": 108, "failed": 0, "records": records}), encoding="utf-8")
    result = import_official(tmp_path)
    assert result["endpoint"] == "official_fulltrack_confidence"
    assert all(row["metric_family"] == "official_fulltrack" for row in result["rows"])
    assert all(row["endpoint"] == "confidence" for row in result["rows"])


def test_blind_subject_manifest_has_no_condition_labels(tmp_path: Path) -> None:
    pairs = [{"pair_id": "opaque", "source_group": "g", "sample_id": "s", "tts_arm": "qwen_cloud", "natural_video": "/tmp/natural.mkv", "tts_video": "/tmp/tts.mkv"}]
    result = export_blind_pack(pairs, tmp_path, raters_per_pair=3)
    text = (tmp_path / "subject_trials.csv").read_text(encoding="utf-8")
    assert result["trial_count"] == 3
    assert "qwen_cloud" not in text
    assert "natural" not in text
