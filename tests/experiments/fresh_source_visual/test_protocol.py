from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.fresh_source_visual import common, runner, validate


def _feature(values: np.ndarray, valid: np.ndarray | None = None, times: np.ndarray | None = None) -> common.FeatureArray:
    values = np.asarray(values, dtype=np.float64)
    mouth = np.zeros((len(values), 31, 2), dtype=np.float64)
    mouth[:, 0, 0] = values
    if valid is None:
        valid = np.ones(len(values), dtype=bool)
    if times is None:
        times = np.arange(len(values), dtype=np.float64) / common.FPS
    mouth[~np.asarray(valid, dtype=bool)] = 0.0
    return common.validate_feature_arrays(mouth, np.asarray(valid, dtype=bool), times, metadata={})


def _groups(count: int = 12) -> tuple[list[str], dict[str, common.FeatureArray]]:
    groups = [f"g{i:02d}" for i in range(count)]
    t = np.arange(common.FRAME_COUNT, dtype=np.float64)
    # Nonstationary trajectory makes both integer shifts and reversal visible.
    values = np.sin(t / 7.0) + 0.01 * t + np.cos(t / 13.0)
    return groups, {group: _feature(values) for group in groups}


def test_real_calibration_passes_fixed_shift_and_reverse_gate() -> None:
    groups, features = _groups()
    result = common.run_calibration(features, groups)
    assert result["status"] == "METRIC_CALIBRATED"
    assert result["distortion_counts"] == {"shift_plus5": 12, "shift_minus5": 12, "reverse": 12}
    assert max(row["identity_mse"] or 0.0 for row in result["rows"]) <= 1e-12


def test_calibration_does_not_require_cross_time_pts_for_shift() -> None:
    groups, features = _groups()
    # A valid fixed 25-fps clock naturally has a 200 ms PTS difference at +5.
    pair = common.calibration_pair(features[groups[0]], transform="shift_plus5")
    assert pair["valid_count"] == 90
    assert pair["coverage"] >= 0.90
    assert pair["mse"] > 1e-6


def test_common_support_boundary_keeps_80_missing_and_81_observed() -> None:
    groups, real = _groups(1)
    source = real[groups[0]]
    arms: dict[str, common.FeatureArray] = {"real": source}
    for name in ("N42", "C42", "N43", "C43"):
        arms[name] = source
    support, _info = common.fixed_clock_support(arms, ("real", "N42", "C42", "N43", "C43"))
    assert len(support) == 90
    invalid = source.valid.copy()
    invalid[25:34] = False  # leaves 81 rows in J0
    sparse = _feature(source.mouth[:, 0, 0], valid=invalid)
    arms["C43"] = sparse
    support, _ = common.fixed_clock_support(arms, ("real", "N42", "C42", "N43", "C43"))
    assert len(support) == 81
    invalid[34] = False  # leaves 80 rows: group must be missing, not dropped
    sparse = _feature(source.mouth[:, 0, 0], valid=invalid)
    arms["C43"] = sparse
    support, _ = common.fixed_clock_support(arms, ("real", "N42", "C42", "N43", "C43"))
    assert len(support) == 80


def test_static_dynamic_identity_and_reversal_q() -> None:
    t = np.arange(common.FRAME_COUNT, dtype=np.float64)
    trajectory = t + 0.1 * np.sin(t / 8.0)
    real = _feature(trajectory)
    natural = _feature(trajectory + 0.5)
    candidate = _feature(trajectory)
    values = common.group_decomposition(
        {"real": real, "N42": natural, "N43": natural, "C42": candidate, "C43": candidate},
        common.J0,
    )
    assert values["e_dynamic_natural"] == pytest.approx(0.0, abs=1e-12)
    assert values["e_static_natural"] > 0.0
    assert values["q_candidate"] > 0.0


def test_blind_package_has_two_anonymous_reviewer_packages_and_pending_humans(tmp_path: Path) -> None:
    groups = [f"g{i:02d}" for i in range(12)]
    records = {group: {"sample_id": f"sample_{i:02d}", "real_video": str(tmp_path / "missing.mp4")} for i, group in enumerate(groups)}
    package = runner.build_blind_package(tmp_path, groups, records, tmp_path / "run/B")
    assert package["primary_trials"] == 24
    assert package["qc_trials"] == 4
    assert package["human_status"] == "pending"
    assert package["visual_verified"] is False
    secret = Path(package["secret_mapping_path"])
    assert secret.is_file()
    for reviewer in (1, 2):
        public = json.loads((tmp_path / f"run/B/blind/reviewer_{reviewer}/package.json").read_text())
        assert public["primary_trials"] == 24
        assert public["qc_trials"] == 4
        assert all(not {"model", "source_group", "sample_id", "condition", "seed"}.intersection(row) for row in public["trials"])


def test_independent_comparator_rejects_resigned_values() -> None:
    assert validate._json_equal({"value": 1.0}, {"value": 1.0}) == []
    assert validate._json_equal({"value": 1.0}, {"value": 2.0})


def test_synthetic_complete_branch_round_trips_independent_validator(tmp_path: Path) -> None:
    groups, real_features = _groups()
    run_root = tmp_path
    formal = [{"source_group": group} for group in groups]
    asset_path = tmp_path / "face_landmarker.task"
    asset_path.write_bytes(b"synthetic-landmarker")
    asset = common.media_metadata(asset_path)
    records = []
    for index, group in enumerate(groups):
        real_path = tmp_path / f"{group}.mkv"
        real_path.write_bytes(f"synthetic-real-{group}".encode())
        records.append({"source_group": group, "sample_id": f"sample_{index:02d}", "real_video": str(real_path), "real_video_sha256": common.file_sha256(real_path)})
    common.write_json(run_root / "cohort.json", {"schema_version": 1, "status": "GO", "readiness": "COHORT_READY", "formal": formal, "smoke": []})
    common.write_json(run_root / "inputs.json", {"schema_version": 1, "status": "GO", "records": records})
    branch = run_root / "run/B"
    index: dict[str, object] = {"schema_version": 1, "fixed_clock_frames": 140, "fps": 25.0, "asset": asset, "rows": [], "generated": []}
    for group, record in zip(groups, records, strict=True):
        for model in common.MODELS:
            for arm in ("N42", "C42", "N43", "C43"):
                path = runner._feature_path(branch, str(record["sample_id"]), f"{model}_{arm}")
                source = real_features[group]
                common.save_feature(path, common.FeatureArray(source.mouth, source.valid, source.timestamps, {**asset, "source_group": group, "source_kind": f"generated:{model}"}))
                index["generated"].append({"sample_id": record["sample_id"], "source_group": group, "model": model, "arm": arm, "status": "complete", "path": str(path), "sha256": common.file_sha256(path)})
            path = runner._feature_path(branch, str(record["sample_id"]), "R")
            common.save_feature(path, common.FeatureArray(real_features[group].mouth, real_features[group].valid, real_features[group].timestamps, {**asset, "source_group": group, "source_kind": "real", "video_path": str(Path(str(record["real_video"])).resolve()), "video_sha256": record["real_video_sha256"]}))
            index["rows"].append({"sample_id": record["sample_id"], "source_group": group, "arm": "R", "path": str(path), "sha256": common.file_sha256(path)})
    common.write_json(branch / "features.json", index)
    calibration = common.run_calibration(real_features, groups)
    common.write_json(branch / "calibration.json", calibration)
    assert runner.analyse(run_root) == 0
    assert runner.blind(run_root) == 0
    result = validate.validate_run(run_root)
    assert result["status"] == "GO", result

    analysis_path = branch / "analysis.json"
    tampered = json.loads(analysis_path.read_text())
    tampered["models"]["wav2lip"]["status"] = "SIGNAL"
    common.write_json(analysis_path, tampered)
    result = validate.validate_run(run_root)
    assert result["status"] == "NO_GO"
