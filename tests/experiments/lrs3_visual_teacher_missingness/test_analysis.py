from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.lrs3_visual_teacher_missingness import analysis, runner
from scripts.experiments.lrs3_visual_teacher_missingness import (
    validate as independent_validate,
)
from scripts.experiments.lrs3_visual_teacher_missingness.common import (
    ProtocolError,
    write_json,
)


def feature(times, values, valid=None):
    mouth = np.zeros((len(times), 31, 2), dtype=np.float64)
    mouth[:, 0, 0] = values
    return {
        "mouth": mouth,
        "valid": np.ones(len(times), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool),
        "timestamps": np.asarray(times, dtype=np.float64),
    }


def test_common_support_uses_same_real_indices_and_tail_denominator():
    real = feature([0.0, 0.04, 0.08, 0.12], [0, 1, 2, 3])
    natural = feature([0.0, 0.04], [0, 1])
    candidate = feature([0.0, 0.04, 0.08], [0, 1, 2])
    gi, ni, mi = analysis._common_support(
        {"real": real, "natural": natural, "candidate": candidate}
    )
    assert gi.tolist() == [0, 1]
    assert ni.tolist() == [0, 1]
    assert mi.tolist() == [0, 1]


def test_invalid_cached_rows_are_not_silently_missing():
    real = feature([0.0, 0.04], [0, 0], valid=[True, False])
    natural = feature([0.0, 0.04], [0, 0])
    candidate = feature([0.0, 0.04], [0, 0])
    exact = {
        "natural": analysis.exact_metric(real, natural),
        "candidate": analysis.exact_metric(real, candidate),
    }
    ok, checks = analysis._old_eligibility(
        {"eligibility": {"eligible": True}},
        exact,
        {"real": real, "natural": natural, "candidate": candidate},
    )
    assert not ok
    assert not checks["real_valid_fraction"]


def test_bounds_keep_all_groups_and_are_equal_weighted():
    rows = [
        {"source_group": "g0", "observed": True, "b": 0.5, "missing_reasons": []},
        {"source_group": "g0", "observed": False, "b": None, "missing_reasons": ["x"]},
        {"source_group": "g1", "observed": True, "b": -0.5, "missing_reasons": []},
    ]
    groups, lower, upper = analysis.group_bounds(rows, ["g0", "g1", "g2"])
    assert groups[-1]["L_g"] == -1.0 and groups[-1]["U_g"] == 1.0
    assert lower == np.mean([(-0.5) / 2, -0.5, -1.0])
    assert upper == np.mean([(1.5) / 2, -0.5, 1.0])


def test_static_dynamic_mse_identity_and_constant_bias():
    real = feature([0.0, 0.04, 0.08], [0.0, 1.0, 0.0])
    natural = feature([0.0, 0.04, 0.08], [0.5, 1.5, 0.5])
    candidate = feature([0.0, 0.04, 0.08], [0.0, 1.0, 0.0])
    values = analysis._decomposition(
        {"real": real, "natural": natural, "candidate": candidate},
        np.arange(3),
        np.arange(3),
        np.arange(3),
    )
    assert values["e_dynamic_natural"] == pytest.approx(0.0)
    assert values["e_total_natural"] == pytest.approx(values["e_static_natural"])
    assert values["identity_error_natural"] == pytest.approx(0.0, abs=1e-12)
    assert values["e_dynamic_candidate"] == pytest.approx(0.0)


def test_reversal_control_detects_order_but_static_sequence_is_retained():
    real = feature([0.0, 0.04, 0.08, 0.12], [0.0, 1.0, 2.0, 3.0])
    reversed_arm = feature([0.0, 0.04, 0.08, 0.12], [3.0, 2.0, 1.0, 0.0])
    values = analysis._decomposition(
        {"real": real, "natural": reversed_arm, "candidate": reversed_arm},
        np.arange(4),
        np.arange(4),
        np.arange(4),
    )
    assert values["e_static_natural"] == pytest.approx(0.0)
    assert values["e_dynamic_natural"] > 0
    assert values["q_natural"] < 0


def test_existing_output_requires_explicit_resume(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "marker").write_text("x")
    monkeypatch.setattr(runner.config, "run_root_for", lambda _: root)
    assert runner.run("existing", "prepare", resume=False) == 1
    assert "requires --resume" in (root / "error.json").read_text()


def test_run_root_matches_implementation_package_name():
    assert (
        runner.config.run_root_for("smoke").name
        == "lrs3_visual_teacher_missingness_smoke"
    )


def test_cached_support_mismatch_is_blocked(tmp_path, monkeypatch):
    cached = tmp_path / "support_indices.npz"
    np.savez(
        cached,
        offsets=np.array([0, 1]),
        real=np.array([0]),
        natural=np.array([0]),
        candidate=np.array([0]),
    )
    monkeypatch.setattr(runner.config, "H", tmp_path)
    with pytest.raises(ProtocolError, match="support indices differ"):
        runner._assert_cached_support(
            {
                "offsets": np.array([0, 1]),
                "real": np.array([1]),
                "natural": np.array([0]),
                "candidate": np.array([0]),
            }
        )


def test_resume_rejects_changed_code_identity(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    write_json(
        root / "protocol.json",
        {
            "code_sha256": {"runner.py": "changed"},
            "spec_sha256": {},
            "cached_h": {"hashes": runner.config.H_HASHES},
            "record_count": 0,
            "group_count": 0,
        },
    )
    write_json(root / "input_audit.json", {})
    write_json(root / "records.json", {"records": [], "groups": [], "split": {}})
    monkeypatch.setattr(runner.config, "run_root_for", lambda _: root)
    monkeypatch.setattr(
        runner,
        "_check_fixed_inputs",
        lambda: ({"artifact_sha256": "h"}, {}, {}, [], [], {}),
    )
    with pytest.raises(ProtocolError, match="resume identity changed"):
        runner.prepare(runner.config.RunPaths(root))


def test_resigned_analysis_cannot_hide_tampered_feature(tmp_path, monkeypatch):
    feature_path = tmp_path / "feature.npz"
    np.savez(
        feature_path,
        canonical_mouth=np.zeros((1, 31, 2)),
        valid=np.ones(1, bool),
        timestamps_s=np.array([0.0]),
    )
    digest = independent_validate.file_sha256(feature_path)
    audit_rows = [{"path": "feature.npz", "sha256": digest}] + [
        {"path": f"unused-{i}.npz", "sha256": "unused"} for i in range(398)
    ]
    record = {
        "feature_paths": {
            "real": str(feature_path),
            "natural": str(feature_path),
            "candidate": str(feature_path),
        }
    }
    monkeypatch.setattr(independent_validate.config, "REPO", tmp_path)
    feature_path.write_bytes(feature_path.read_bytes().replace(b"\x00", b"\x01", 1))
    with pytest.raises(ProtocolError, match="feature binding changed"):
        independent_validate._verify_feature_bindings(
            [record], {"features": audit_rows}
        )


def test_frozen_h_cohort_cannot_be_reassigned_after_resigning() -> None:
    records = [{"sample_id": "s0", "source_group": "g0"}]
    groups = ["g0"]
    cached = {"records": records, "groups": groups}
    independent_validate._assert_frozen_cohort(records, groups, cached)
    with pytest.raises(ProtocolError, match="frozen H records"):
        independent_validate._assert_frozen_cohort(
            [{"sample_id": "s0", "source_group": "other"}],
            ["other"],
            cached,
        )
