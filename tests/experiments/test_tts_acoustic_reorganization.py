"""CPU contract tests for the acoustic-reorganization intervention protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "experiments" / "tts_acoustic_reorganization.py"
SPEC = importlib.util.spec_from_file_location("tts_acoustic_reorganization", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _token_fixture() -> list[dict[str, object]]:
    return [
        {"token": "sil", "start_s": 0.0, "end_s": 0.02, "is_silence": True},
        {"token": "a", "start_s": 0.02, "end_s": 0.16},
        {"token": "sil", "start_s": 0.16, "end_s": 0.22, "is_silence": True},
    ]


def test_interpolation_recenters_and_preserves_endpoints() -> None:
    values = np.arange(5 * 4, dtype=np.float64).reshape(5, 4)
    result = MODULE._center_and_interpolate(values, 8)
    np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(result[0], values[0] - values.mean(axis=0))
    np.testing.assert_allclose(result[-1], values[-1] - values.mean(axis=0))
    np.testing.assert_allclose(MODULE._template_to_length(result, 5).mean(axis=0), 0.0, atol=1e-12)


def test_projection_is_orthogonal_and_reconstructs_centered_signal() -> None:
    q = np.zeros((7, MODULE.FEATURE_DIM), dtype=np.float64)
    q[:, 0] = np.arange(7) - 3
    r = np.zeros_like(q)
    r[:, 1] = np.asarray([-2, 1, 0, 3, -1, 0, -1], dtype=np.float64)
    values = q * 2.5 + r + np.linspace(-4, 4, 7)[:, None]
    components = MODULE._project_components(values, q)
    np.testing.assert_allclose(components["U"], components["P"] + components["R"], atol=1e-12)
    assert abs(float(np.sum(components["P"] * components["R"]))) < 1e-12
    np.testing.assert_allclose(np.asarray(components["P"]).mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(components["R"]).mean(axis=0), 0.0, atol=1e-12)
    assert float(components["p_norm"]) > 0 and float(components["r_norm"]) > 0


def test_interventions_have_equal_budget_mean_endpoints_and_noop_for_zero_component() -> None:
    rng = np.random.default_rng(4)
    z_n = rng.normal(size=(12, MODULE.FEATURE_DIM)).astype(np.float64)
    z_t = rng.normal(size=(12, MODULE.FEATURE_DIM)).astype(np.float64)
    occurrences = [{"occurrence_index": 0, "label": "a", "natural_frame_indices": list(range(2, 10)), "eligible": True, "is_silence": False}]
    q = np.zeros((8, MODULE.FEATURE_DIM), dtype=np.float64); q[:, 0] = np.linspace(-1, 1, 8); q[:, 1] = np.linspace(1, -1, 8)
    arrays, operations, meta = MODULE._intervention_arrays(z_n, z_t, occurrences, {"a": q})
    assert meta["active_occurrence_count"] == 1
    op = operations[0]; delta = float(op["delta"])
    for arm, source, component in (("N_R_DOWN", "N_ID", "r_norm_n"), ("N_P_DOWN", "N_ID", "p_norm_n"), ("T_R_DOWN", "T_ID", "r_norm_t"), ("T_P_DOWN", "T_ID", "p_norm_t")):
        assert np.linalg.norm(arrays[arm][2:10] - arrays[source][2:10]) == pytest.approx(delta, rel=1e-8, abs=1e-8)
        np.testing.assert_allclose(arrays[arm][2:10].mean(axis=0), arrays[source][2:10].mean(axis=0), atol=1e-10)
        np.testing.assert_array_equal(arrays[arm][:2], arrays[source][:2]); np.testing.assert_array_equal(arrays[arm][10:], arrays[source][10:])
        assert 0.5 - 1e-12 <= 1.0 - delta / float(op[component]) <= 1.0 + 1e-12
    z = np.zeros((12, MODULE.FEATURE_DIM), dtype=np.float64); z[2:10, 0] = np.linspace(-1, 1, 8)
    q_zero = np.zeros((8, MODULE.FEATURE_DIM), dtype=np.float64); q_zero[:, 0] = np.linspace(-1, 1, 8)
    arrays_zero, operations_zero, meta_zero = MODULE._intervention_arrays(z, z, occurrences, {"a": q_zero})
    assert meta_zero["active_occurrence_count"] == 0
    assert operations_zero[0]["reason"] == "zero_component"
    for key in ("N_R_DOWN", "N_P_DOWN", "T_R_DOWN", "T_P_DOWN"):
        np.testing.assert_array_equal(arrays_zero[key], z)


def test_parent_mask_uses_natural_clock_when_tts_duration_differs() -> None:
    tokens_n = _token_fixture()
    tokens_t = [
        {"token": "sil", "start_s": 0.0, "end_s": 0.01, "is_silence": True},
        {"token": "a", "start_s": 0.01, "end_s": 0.10},
        {"token": "sil", "start_s": 0.10, "end_s": 0.13, "is_silence": True},
    ]
    parent = [{"occurrence_index": 1, "eligible": True}]
    rows = MODULE._parent_occurrences(parent, tokens_n, tokens_t, 10)
    assert rows[1]["eligible"] is True
    assert rows[1]["natural_frame_indices"] == rows[1]["tts_frame_indices"]


def test_template_builder_equal_weights_and_donor_only() -> None:
    rows = []
    raw = {}
    for sid in range(1, 6):
        z = np.zeros((10, MODULE.FEATURE_DIM), dtype=np.float32); z[2:8, 0] = sid; z[2:8, 1] = np.arange(6)
        rows.append({"sample_id": sid, "paired_key": f"donor-{sid}", "natural_tokens": _token_fixture(), "tts_tokens": _token_fixture(), "z_n": z, "z_t": z})
        raw[sid] = z.copy()
    templates, meta = MODULE._template_build(rows, raw)
    assert "a" in templates
    assert meta["templates"]["a"]["donor_paired_keys"] == [f"donor-{sid}" for sid in range(1, 6)]
    np.testing.assert_allclose(templates["a"].mean(axis=0), 0.0, atol=1e-12)
    assert {item["sample_id"] for item in meta["occurrences"]} <= set(range(1, 6))


def test_expected_21_cells_and_curve_metrics() -> None:
    cells = MODULE.expected_cells([6])
    assert len(cells) == 21
    assert (6, "N_RAW", "N_RAW") in cells and (6, "T_P_DOWN", "T_ID") in cells
    assert len({(video, audio) for _, video, audio in cells}) == 21
    visual = np.zeros((60, 4), dtype=np.float32); audio = np.zeros((60, 4), dtype=np.float32)
    indices = list(range(15, 45)); lags, values = MODULE._curve(visual, audio, indices)
    metrics = MODULE._curve_metrics(lags, values)
    assert metrics["k_star"] == -15
    assert metrics["C"] == pytest.approx(0.0)


def test_c_delta_decomposition_matches_c_definition() -> None:
    old = {"C": 0.25, "B": 1.10, "D": 0.85}
    new = {"C": 0.40, "B": 1.00, "D": 0.60}
    decomposition = MODULE._c_delta_decomposition(new, old)
    assert decomposition["delta_C"] == pytest.approx(0.15)
    assert decomposition["B_new_minus_old"] == pytest.approx(-0.10)
    assert decomposition["D_old_minus_new"] == pytest.approx(0.25)
    assert decomposition["identity_error"] == pytest.approx(0.0)


def test_bootstrap_is_shared_and_exactly_seeded() -> None:
    first = MODULE._bootstrap_indices(10)
    second = MODULE._bootstrap_indices(10)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (MODULE.BOOTSTRAP_DRAWS, 10)
    values = np.arange(10, dtype=np.float64)
    summary = MODULE._stat_summary(values, first)
    assert summary["n"] == 10
    assert summary["ci_bonferroni_coverage"] == pytest.approx(1.0 - 0.05 / 6.0)


def test_exact_length_rejects_out_of_range_and_records_right_adjustment() -> None:
    values = np.zeros(20, dtype=np.float32)
    adjusted, metadata = MODULE._exact_length(values, 10)
    assert adjusted.shape == (10,)
    assert metadata["action"] == "right_crop" and metadata["adjustment_sample_count"] == 10
    with pytest.raises(MODULE.ProtocolError):
        MODULE._write_generated_audio(Path("/tmp/unused-float.wav"), Path("/tmp/unused-pcm.wav"), np.array([1.2], dtype=np.float32), 1)
