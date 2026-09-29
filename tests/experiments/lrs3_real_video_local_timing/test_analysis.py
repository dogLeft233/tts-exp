from pathlib import Path

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing import config
from scripts.experiments.lrs3_real_video_local_timing.analysis import analyze
from scripts.experiments.lrs3_real_video_local_timing.media import build_mapping
from scripts.experiments.lrs3_real_video_local_timing.scoring import (
    local_evidence,
    reconstruct_global,
)


def _protocol() -> dict[str, object]:
    mapping = build_mapping(150)
    return {
        "records": [
            {
                "sample_id": f"s{index}",
                "source_group": f"g{index}",
                "crop": {"frame_count": 150, "mappings": mapping},
                "natural_audio_format": {"sample_count": 150 * 640},
            }
            for index in range(config.EXPECTED_RECORD_COUNT)
        ]
    }


def _score_row(tmp_path: Path, protocol_record: dict[str, object], arm: str, *, wrong: bool = False, flat: bool = False) -> dict[str, object]:
    mapping = protocol_record["crop"]["mappings"][config.WARP_ARM]
    matrix = np.full((145, 31), 2.0, dtype=np.float64)
    evidence = local_evidence(matrix, mapping, 150, 150 * 640)
    for name in ("PLUS", "MINUS"):
        target = 15
        if arm == config.WARP_ARM:
            target = 18 if name == "PLUS" and not wrong else (12 if name == "MINUS" and not wrong else 15)
        if flat and name == "PLUS":
            continue
        for row in evidence[name]["rows"]:
            matrix[row, target] = 0.0
    path = tmp_path / f"{protocol_record['sample_id']}_{arm}.npy"
    np.save(path, matrix, allow_pickle=False)
    reconstructed = reconstruct_global(matrix)
    local = local_evidence(matrix, mapping, 150, 150 * 640)
    return {
        "sample_id": protocol_record["sample_id"],
        "arm": arm,
        "matrix": str(path),
        "reconstructed": reconstructed,
        "local": local,
    }


def _manifest(tmp_path: Path, *, wrong_count: int = 0, flat_count: int = 0, repeat_bad: bool = False) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = _protocol()
    rows: list[dict[str, object]] = []
    for index, record in enumerate(protocol["records"]):
        for arm in config.ARMS:
            row = _score_row(tmp_path, record, arm, wrong=arm == config.WARP_ARM and index < wrong_count, flat=arm in (config.REAL_ARM, config.REPEAT_ARM) and index < flat_count)
            if repeat_bad and arm == config.REPEAT_ARM and index == 0:
                row["matrix"] = row["matrix"]
                value = np.load(row["matrix"], allow_pickle=False)
                value[20, 0] = 9.0
                np.save(row["matrix"], value, allow_pickle=False)
                row["reconstructed"] = reconstruct_global(value)
                row["local"] = local_evidence(value, record["crop"]["mappings"][config.WARP_ARM], 150, 150 * 640)
            rows.append(row)
    return {"status": "complete", "scores": rows}


def test_local_timing_decision_uses_18_of_22_boundary(tmp_path: Path) -> None:
    result = analyze(_protocol(), _manifest(tmp_path, wrong_count=4))
    assert result["scientific_decision"] == "LOCAL_TIMING_DETECTED"
    assert result["recovery"]["success_count"] == 18

    result = analyze(_protocol(), _manifest(tmp_path / "seventeen", wrong_count=5))
    assert result["scientific_decision"] == "LOCAL_TIMING_NOT_ESTABLISHED"
    assert result["recovery"]["success_count"] == 17


def test_repeatability_precedes_baseline_and_recovery(tmp_path: Path) -> None:
    result = analyze(_protocol(), _manifest(tmp_path, repeat_bad=True))
    assert result["scientific_decision"] == "REPEATABILITY_FAILED"


def test_baseline_inconclusive_keeps_fixed_22_record_denominator(tmp_path: Path) -> None:
    result = analyze(_protocol(), _manifest(tmp_path, flat_count=3))
    assert result["scientific_decision"] == "BASELINE_INCONCLUSIVE"
    assert result["record_count"] == 22
    assert result["baseline"]["clear_and_aligned_count"] == 19
