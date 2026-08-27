from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "scripts/experiments/lrs3_tts_visual_control/run_visual_teacher_audit.py"
SPEC = importlib.util.spec_from_file_location("lrs3_tts_visual_teacher_audit", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from video_features import MOUTH_INDICES, VisualSequence, mouth_features  # noqa: E402


def make_sequence(frame_count: int = 8) -> VisualSequence:
    landmarks = np.zeros((frame_count, 478, 2), dtype=np.float64)
    for index in range(frame_count):
        openness = 0.02 + 0.01 * index
        landmarks[index, 33] = [0.0, 0.0]
        landmarks[index, 263] = [2.0, 0.0]
        landmarks[index, 1] = [1.0, 0.5]
        landmarks[index, MOUTH_INDICES] = np.array([
            [0.35, -0.25], [0.45, -0.24], [0.55, -0.23], [0.65, -0.22],
            [0.75, -0.21], [1.00, -0.20], [1.25, -0.21], [1.35, -0.22],
            [1.45, -0.23], [1.55, -0.24], [1.65, -0.25], [1.70, -0.26],
            [1.60, 0.28], [1.45, 0.29], [1.30, 0.30], [1.15, 0.31],
            [1.00, openness], [0.85, 0.31], [0.70, 0.30], [0.55, 0.29],
            [0.40, 0.28], [0.30, 0.27], [0.25, 0.10], [0.35, 0.05],
            [0.55, 0.02], [1.00, 0.01], [1.45, 0.02], [1.65, 0.05],
            [1.75, 0.10], [1.70, -0.10], [1.65, -0.20],
        ])
    features = mouth_features(landmarks)
    return VisualSequence(
        landmarks=landmarks,
        canonical_mouth=features["canonical_mouth"],
        valid=features["valid"],
        mouth_open=features["mouth_open"],
        mouth_width=features["mouth_width"],
        mouth_height=features["mouth_height"],
        mouth_area=features["mouth_area"],
        timestamps_s=np.arange(frame_count, dtype=np.float64) / 25.0,
        metadata={"synthetic": True, "schema_version": 1},
    )


def metric_row(group: str, exact: float, dtw: float) -> dict:
    return {
        "source_group": group,
        "eligibility": {"eligible": True},
        "metrics": {
            "exact_time": {"benefit": exact},
            "visual_dtw": {"benefit": dtw},
        },
    }


def test_json_safe_converts_nonfinite_values_to_null():
    result = runner._json_safe({"nan": float("nan"), "inf": float("inf"), "finite": np.float64(1.5)})
    assert result == {"nan": None, "inf": None, "finite": 1.5}


def test_metric_gate_requires_all_primary_and_secondary_contracts():
    sequence = make_sequence()
    metrics = {
        "exact_time": {
            "natural": {"distance": 1.0, "coverage": 0.9},
            "candidate": {"distance": 0.5, "coverage": 0.9},
            "benefit": 0.5,
        },
        "visual_dtw": {
            "natural": {"distance": 1.0, "coverage": 0.9, "path_length": 8, "warp_burden": 0.0},
            "candidate": {"distance": 0.5, "coverage": 0.9, "path_length": 8, "warp_burden": 0.0},
            "benefit": 0.5,
        },
    }
    passed = runner._metric_gate({"real": sequence, "natural": sequence, "candidate": sequence}, metrics)
    assert passed["eligible"] is True

    metrics["exact_time"]["candidate"]["coverage"] = 0.84
    rejected = runner._metric_gate({"real": sequence, "natural": sequence, "candidate": sequence}, metrics)
    assert rejected["eligible"] is False
    assert rejected["checks"]["exact_time_coverage_at_least_0_85"] is False


def test_exact_sign_flip_enumerates_all_signs():
    result = runner._exact_sign_flip([1.0, 1.0])
    assert result["n_groups"] == 2
    assert result["enumerated_signs"] == 4
    assert result["mean"] == 1.0
    assert result["p_value"] == pytest.approx(0.25)


def test_statistics_requires_every_effective_group():
    incomplete = runner._statistics([metric_row("g0", 1.0, 0.5)], ["g0", "g1"])
    assert incomplete["complete"] is False
    assert incomplete["missing_eligible_groups"] == ["g1"]

    complete = runner._statistics(
        [metric_row("g0", 1.0, 0.5), metric_row("g1", 2.0, 1.0)],
        ["g0", "g1"],
    )
    assert complete["complete"] is True
    assert complete["group_statistics"]["group_count"] == 2
    assert complete["primary"]["sign_flip"]["enumerated_signs"] == 4


def test_decision_separates_engineering_block_from_scientific_gate():
    blocked = runner._decision({"complete": False}, eligible_count=20, minimum_count=30)
    assert blocked == ("BLOCKED", "not_available", "eligible denominator or effective-group coverage is incomplete")

    no_go = runner._decision(
        {"complete": True, "primary": {"sign_flip": {"mean": -1.0, "p_value": 1.0}}},
        eligible_count=30,
        minimum_count=30,
    )
    assert no_go[0:2] == ("GO", "NO_GO")

    go = runner._decision(
        {"complete": True, "primary": {"sign_flip": {"mean": 1.0, "p_value": 0.01}}},
        eligible_count=30,
        minimum_count=30,
    )
    assert go[0:2] == ("GO", "GO")


def test_record_result_writes_three_feature_snapshots(tmp_path: Path, monkeypatch):
    sequence = make_sequence()
    videos = {}
    for arm in ("real", "natural", "candidate"):
        path = tmp_path / f"{arm}.mp4"
        path.write_bytes(arm.encode("utf-8"))
        videos[arm] = path

    monkeypatch.setattr(runner, "extract_video_features", lambda path, asset: sequence)
    record = {
        "sample_id": "synthetic_00001",
        "source_group": "g0",
        "real_video": str(videos["real"]),
        "real_video_sha256": runner.file_sha256(videos["real"]),
        "natural_video": str(videos["natural"]),
        "natural_video_sha256": runner.file_sha256(videos["natural"]),
        "candidate_video": str(videos["candidate"]),
        "candidate_video_sha256": runner.file_sha256(videos["candidate"]),
    }
    asset = tmp_path / "asset.task"
    asset.write_bytes(b"asset")
    result = runner._record_result(record, asset, tmp_path / "features")

    assert result["eligibility"]["eligible"] is True
    assert set(result["feature_paths"]) == {"real", "natural", "candidate"}
    assert all(Path(path).is_file() for path in result["feature_paths"].values())
    assert result["metrics"]["exact_time"]["benefit"] == pytest.approx(0.0)


def test_record_extraction_reuses_one_landmarker_handle(tmp_path: Path, monkeypatch):
    handles = []

    class FakeLandmarker:
        closed = 0

        def close(self):
            self.closed += 1

    landmarker = FakeLandmarker()
    monkeypatch.setattr(runner, "create_landmarker", lambda _: (object(), landmarker))

    def fake_record(record, asset_snapshot, feature_root, landmarker_handle):
        handles.append(landmarker_handle)
        return {"sample_id": record["sample_id"], "source_group": record["source_group"]}

    monkeypatch.setattr(runner, "_record_result", fake_record)
    records = [
        {"sample_id": "s0", "source_group": "g0"},
        {"sample_id": "s1", "source_group": "g1"},
    ]
    rows, failures = runner._extract_records(records, tmp_path / "asset.task", tmp_path / "features", tmp_path / "records")

    assert failures == []
    assert len(rows) == 2
    assert handles[0] is handles[1]
    assert landmarker.closed == 1


def test_stage_id_rejects_unregistered_retry_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="unexpected Stage01 output directory"):
        runner._stage_id(tmp_path / "01_visual_teacher_audit")


def test_extract_records_rejects_nonpositive_workers(tmp_path: Path):
    with pytest.raises(ValueError, match="workers must be positive"):
        runner._extract_records([], tmp_path / "asset.task", tmp_path / "features", tmp_path / "records", workers=0)


def test_checked_in_review_binds_all_cli_sources():
    source_paths = (
        MODULE_PATH,
        REPO / "scripts/experiments/lrs3_tts_visual_advantage/video_features.py",
        REPO / "scripts/experiments/lrs3_tts_visual_advantage/visual_metrics.py",
        Path(__file__).resolve(),
        REPO / "tests/experiments/lrs3_tts_visual_advantage/test_visual_features_metrics.py",
    )
    review_path = REPO / "_reviews/lrs3_tts_visual_control/01_visual_teacher_audit/code_review.json"
    assert runner._validate_review(review_path, source_paths)["status"] == "PASS"


def test_review_binding_rejects_changed_source(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    source_manifest = [{"path": str(source.resolve()), "sha256": runner.file_sha256(source)}]
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({
            "stage_id": "01_visual_teacher_audit",
            "status": "PASS",
            "test_status": "PASS",
            "source_manifest_sha256": runner.canonical_sha256(source_manifest),
        }),
        encoding="utf-8",
    )
    assert runner._validate_review(review, (source,))["status"] == "PASS"
    source.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        runner._validate_review(review, (source,))
