from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.portable_source_axis_20260930 import ops
from scripts.experiments.portable_source_axis_20260930.independent_consumer import (
    _random_arm_dose_eligible,
)
from scripts.experiments.portable_source_axis_20260930.independent_validator import (
    CONDITIONS,
    ValidationError,
    bootstrap_speaker_mean,
    curve_summary,
    reduce_calibration_anchor,
    reduce_two_clip_speaker_effects,
    recompute_distance_matrix,
    recompute_four_cell_summary,
    row_unit_geometry,
    validate_common_support,
    validate_cross_embedding_artifacts,
    validate_intervention,
    validate_random_dose,
    validate_random_direction,
    validate_strict_projection,
)


def _latent_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.full((3, ops.LATENT_DIM), 0.4, dtype=np.float32)
    z[1, 0:4] = np.asarray([0.01, 0.04, 0.20, 0.8], dtype=np.float32)
    mask = np.asarray([True, True, False])
    u = np.zeros(ops.LATENT_DIM, dtype=np.float64)
    u[3] = 1.0
    d = np.zeros(ops.LATENT_DIM, dtype=np.float64)
    d[0:4] = np.asarray([0.25, -0.12, 0.20, 0.15], dtype=np.float64)
    return z, mask, d, u


def test_independent_kkt_recomputes_projection_and_float32_bound() -> None:
    z = np.asarray([0.02, 0.15, 0.42, 0.1], dtype=np.float64)
    target = np.asarray([-0.08, 0.30, 0.60, 0.45], dtype=np.float64)
    u = np.asarray([0.4, -0.2, 0.3, 1.0], dtype=np.float64)
    projected, detail = ops.level_orthogonal_projection(z, target, u)
    check = validate_strict_projection(z, target, u, projected.astype(np.float32))

    assert detail["converged"] is True
    assert check["valid"] is True
    assert check["residual64"] < 2e-12
    assert check["float32_constraint_abs"] <= (
        abs(check["residual64"]) + check["float32_rounding_constraint_abs_bound"] + check["float32_rounding_slack"]
    )
    assert check["active_positive_coordinates"] == 3


def test_strict_intervention_requires_inactive_bitwise_copy() -> None:
    z, mask, d, u = _latent_fixture()
    output, _ = ops.apply_source_direction(z, mask, d, u, sign=1, level_orthogonal=True)
    report = validate_intervention(z, mask, output, d, u, sign=1, level_orthogonal=True)
    assert report["valid"] is True
    assert report["inactive_exact"] is True
    assert report["strict_projection_rows"] == 2

    broken = output.copy()
    broken[2, 0] = np.nextafter(broken[2, 0], np.float32(1.0))
    with pytest.raises(ValidationError, match="inactive latent frames"):
        validate_intervention(z, mask, broken, d, u, sign=1, level_orthogonal=True)


def test_full_intervention_is_exact_relu_shift() -> None:
    z, mask, d, u = _latent_fixture()
    output, _ = ops.apply_source_direction(z, mask, d, u, sign=-1, level_orthogonal=False)
    report = validate_intervention(z, mask, output, d, u, sign=-1, level_orthogonal=False)
    assert report["valid"] is True
    assert report["strict_projection_rows"] == 0


def test_random_match_is_recomputed_after_float32_cast() -> None:
    z = np.full((2, ops.LATENT_DIM), 0.4, dtype=np.float32)
    mask = np.asarray([True, True])
    d = np.zeros(ops.LATENT_DIM, dtype=np.float64)
    d[0:2] = [0.2, -0.1]
    u = np.zeros(ops.LATENT_DIM, dtype=np.float64)
    u[2] = 1.0
    plus, _ = ops.apply_source_direction(z, mask, d, u, sign=1, level_orthogonal=True)
    directions, _ = ops.random_control_directions(d, u, seeds=(20260930,))
    random, metadata = ops.apply_random_direction(z, mask, directions[0], u, plus)
    report = validate_random_dose(
        z, mask, random, plus, directions[0], u, metadata["per_frame_projection_diagnostics"]
    )
    assert report["valid"] is True
    assert report["infeasible_frames"] == 0
    assert report["postcast_abs_dose_error_max"] <= report["dose_tolerance_max"]

    # A false feasible claim must be caught even when the producer's float64
    # diagnostic was within tolerance before the stored float32 cast.
    broken = random.copy()
    broken[0, 0] += np.float32(1e-3)
    with pytest.raises(ValidationError, match="projected multiplier|post-cast"):
        validate_random_dose(
            z, mask, broken, plus, directions[0], u,
            metadata["per_frame_projection_diagnostics"],
        )


def test_postcast_random_dose_miss_is_reported_per_arm_without_discarding_source_rows() -> None:
    z = np.full((2, ops.LATENT_DIM), 0.4, dtype=np.float32)
    mask = np.asarray([True, True])
    d = np.zeros(ops.LATENT_DIM, dtype=np.float64)
    d[0:2] = [0.2, -0.1]
    u = np.zeros(ops.LATENT_DIM, dtype=np.float64)
    u[2] = 1.0
    plus, _ = ops.apply_source_direction(z, mask, d, u, sign=1, level_orthogonal=True)
    directions, _ = ops.random_control_directions(d, u, seeds=(20260930,))

    # Scale zero is a valid projected random output but deliberately misses
    # the positive source-arm dose. The audit mode must preserve this failure
    # as arm-specific metadata rather than raising before source rows can pass.
    random = z.copy()
    diagnostics = []
    for frame in np.flatnonzero(mask):
        target_norm = float(np.linalg.norm(plus[frame].astype(np.float64) - z[frame].astype(np.float64)))
        diagnostics.append({
            "frame_index": int(frame), "target_norm": target_norm, "actual_norm": 0.0,
            "dose_error": -target_norm, "infeasible": False, "scale": 0.0,
            "kkt_residual64": 0.0, "float32_constraint_abs": 0.0,
            "float32_constraint_abs_bound": 0.0,
        })

    report = validate_random_dose(
        z, mask, random, plus, directions[0], u, diagnostics,
        enforce_dose_tolerance=False,
    )
    assert report["valid"] is True
    assert report["dose_match_eligible"] is False
    assert report["dose_tolerance_miss_count"] == 2
    assert _random_arm_dose_eligible(
        {"interventions": {"random1": {"dose": report}}}, "random1"
    ) is False
    assert _random_arm_dose_eligible({}, "baseline") is True

    with pytest.raises(ValidationError, match="post-cast random displacement"):
        validate_random_dose(z, mask, random, plus, directions[0], u, diagnostics)


def test_random_direction_is_orthogonal_to_both_axes_and_has_frozen_norm() -> None:
    d = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    report = validate_random_direction(np.asarray([0.0, 0.0, 0.5]), d, u)
    assert report["valid"] is True
    assert report["direction_norm"] == pytest.approx(0.5)
    assert report["target_norm"] == pytest.approx(0.5)
    with pytest.raises(ValidationError, match="span\\(d,u\\)"):
        validate_random_direction(np.asarray([0.1, 0.0, np.sqrt(0.24)]), d, u)

    d512 = np.zeros(ops.LATENT_DIM, dtype=np.float64)
    u512 = np.zeros(ops.LATENT_DIM, dtype=np.float64)
    d512[:3] = [0.30, 0.10, -0.05]
    u512[:3] = [0.20, 1.00, 0.15]
    directions, _ = ops.random_control_directions(d512, u512, seeds=(20260930, 20260931))
    for direction in directions:
        assert validate_random_direction(direction, d512, u512)["valid"] is True


def test_support_must_be_the_same_all_eight_arms() -> None:
    base = np.asarray([True, True, False, True])
    masks = {name: base.copy() for name in CONDITIONS}
    report = validate_common_support(masks, declared_common=base)
    assert report == {"support_n": 3, "clock_n": 4, "valid": True}

    masks["random2"][2] = True
    with pytest.raises(ValidationError, match="supports differ"):
        validate_common_support(masks)


def test_lag_column_anchor_sign_and_score_identity() -> None:
    matrix = np.asarray([
        [4.0, 3.0, 2.0, 1.0, 2.0],
        [5.0, 4.0, 3.0, 2.0, 3.0],
    ], dtype=np.float32)
    summary = curve_summary(matrix, anchor_column=4)
    assert summary["C"] == summary["B"] - summary["D"]
    assert summary["anchor_k"] == 2
    assert summary["anchor_official_offset"] == -2
    assert summary["C_anchor"] == summary["B"] - summary["D_anchor"]


def test_calibration_anchor_uses_q10_clip_then_speaker_medians() -> None:
    rows = []
    for speaker_index in range(10):
        k = 0 if speaker_index < 5 else 1
        matrix = np.full((2, 31), 10.0, dtype=np.float32)
        matrix[:, 15 + k] = 0.0
        for clip_index in range(2):
            rows.append({
                "sample_id": f"S{speaker_index:02}_{clip_index}",
                "speaker_id": f"S{speaker_index:02}",
                "split": "cal",
                "cell": "q10",
                "matrix": matrix.copy(),
            })
    result = reduce_calibration_anchor(rows)
    assert result["global_median_k"] == 0.5
    # Half tie: k=1 is nearer the historical k=3 than k=0.
    assert result["anchor_k"] == 1
    assert result["anchor_column"] == 16
    assert result["official_offset"] == -1

    rows[0]["cell"] = "q00"
    with pytest.raises(ValidationError, match="calibration q10"):
        reduce_calibration_anchor(rows)


def test_numpy_distance_frontend_retains_syncnet_epsilon() -> None:
    identical = np.zeros((4, 3), dtype=np.float32)
    matrix = recompute_distance_matrix(identical, identical, vshift=1)
    expected_equal = np.sqrt(np.float32(3) * np.float32(1e-12))
    assert matrix.shape == (4, 3)
    assert np.all(matrix[1:3, 1] > 0)
    assert np.isclose(matrix[1, 1], expected_equal, atol=1e-12)
    assert matrix.dtype == np.float32


def test_full_cross_distance_audit_rebuilds_raw_unit_and_zero_arrays() -> None:
    lip = np.asarray([[0.2, -0.3, 0.5], [0.9, 0.1, -0.4]], dtype=np.float32)
    audio = np.asarray([[0.1, -0.2, 0.6], [-0.5, 0.4, 0.2], [0.0, 0.0, 0.0]], dtype=np.float32)

    def pairwise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.empty((len(x), len(y)), dtype=np.float32)
        for i, left in enumerate(x):
            for j, right in enumerate(y):
                difference = (left - right).astype(np.float32)
                difference += np.float32(1e-6)
                out[i, j] = np.sqrt(np.sum(difference * difference, dtype=np.float32)).astype(np.float32)
        return out

    raw = pairwise(lip, audio)
    unit_lip = row_unit_geometry(lip)
    unit_audio = row_unit_geometry(audio)
    unit = pairwise(unit_lip, unit_audio)
    raw_zero = pairwise(lip, np.zeros((1, lip.shape[1]), dtype=np.float32))[:, 0]
    unit_zero = pairwise(unit_lip, np.zeros((1, lip.shape[1]), dtype=np.float32))[:, 0]
    report = validate_cross_embedding_artifacts(lip, audio, raw, unit, raw_zero, unit_zero, block_rows=1)

    assert report["valid"] is True
    assert report["visual_rows"] == 2
    assert report["audio_rows"] == 3
    assert max(value for key, value in report.items() if key.endswith("max_abs_error")) == 0.0

    corrupted = raw.copy()
    corrupted[0, 1] += np.float32(0.01)
    with pytest.raises(ValidationError, match="raw_cross"):
        validate_cross_embedding_artifacts(lip, audio, corrupted, unit, raw_zero, unit_zero)


def test_unit_geometry_and_four_cell_closure_use_identical_support() -> None:
    lip = np.asarray([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    audio = np.asarray([[3.0, 4.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    unit_lip = row_unit_geometry(lip)
    assert np.allclose(np.linalg.norm(unit_lip, axis=1), 1.0)

    # Four score cells deliberately use the same rows. Any score summary must
    # preserve the algebraic G/E/I definitions at both raw and fixed anchor.
    matrices = {
        "q00": recompute_distance_matrix(lip, audio, vshift=1),
        "q10": recompute_distance_matrix(lip * np.float32(1.1), audio, vshift=1),
        "q01": recompute_distance_matrix(lip, audio * np.float32(0.9), vshift=1),
        "q11": recompute_distance_matrix(lip * np.float32(1.1), audio * np.float32(0.9), vshift=1),
    }
    result = recompute_four_cell_summary(matrices, anchor_column=1)
    assert result["valid"] is True
    assert all(cell["support_n"] == 3 for cell in result["cells"].values())
    for metric, values in result["contrasts"].items():
        cells = result["cells"]
        assert values["G"] == cells["q10"][metric] - cells["q00"][metric]
        assert values["E"] == cells["q01"][metric] - cells["q00"][metric]
        assert values["I"] == cells["q11"][metric] - cells["q10"][metric] - cells["q01"][metric] + cells["q00"][metric]


def test_speaker_reducer_and_bootstrap_use_two_clip_equal_speaker_units() -> None:
    rows = [
        {"speaker_id": "A", "effect": 1.0}, {"speaker_id": "A", "effect": 3.0},
        {"speaker_id": "B", "effect": 10.0}, {"speaker_id": "B", "effect": 14.0},
        {"speaker_id": "C", "effect": 100.0}, {"speaker_id": "C", "effect": 120.0},
    ]
    by_speaker = reduce_two_clip_speaker_effects(rows, "effect")
    assert by_speaker == {"A": 2.0, "B": 12.0, "C": 110.0}
    boot = bootstrap_speaker_mean(by_speaker, draws=200, seed=20260930)
    assert boot["mean"] == pytest.approx((2.0 + 12.0 + 110.0) / 3)
    assert boot["unit"] == "speaker"
    assert boot["speaker_count"] == 3

    with pytest.raises(ValidationError, match="exactly two clips"):
        reduce_two_clip_speaker_effects(rows[:-1], "effect")
