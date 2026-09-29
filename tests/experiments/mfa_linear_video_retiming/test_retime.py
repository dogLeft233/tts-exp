from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError
from scripts.experiments.mfa_linear_video_retiming.retime import (
    build_map,
    global_seed,
    knot_positions,
    mirror_map,
    regularization,
    render_map,
    render_nearest,
    validate_map,
)


def test_identity_preserves_every_pixel_and_end_time() -> None:
    source = np.arange(40 * 2 * 3 * 3, dtype=np.uint8).reshape(40, 2, 3, 3)
    mapping = build_map(40, 38)
    output, audit = render_map(source, mapping)

    assert np.array_equal(output, source)
    assert mapping["q"][0] == 0.0
    assert mapping["q"][-1] == 39.0
    assert audit["max_abs_displacement"] == 0.0
    assert audit["regularization"] == 0.0


def test_positive_displacement_reads_later_original_frame_and_rounds_half_up() -> None:
    positions = [5, 17, 29]
    mapping = build_map(40, 40, [0.5, 1.0, 0.0], positions=positions)
    source = np.zeros((40, 1, 1, 3), dtype=np.uint8)
    source[5, 0, 0] = 0
    source[6, 0, 0] = 1
    output, _ = render_map(source, mapping)

    assert mapping["q"][5] == 5.5
    assert output[5, 0, 0, 0] == 1
    assert mapping["q"][6] > 6.0


def test_knots_are_limited_to_sixteen_and_twelve_frames_apart() -> None:
    positions = knot_positions(240, 220)

    assert positions.size <= 16
    assert np.all(np.diff(positions) >= 12)
    assert positions[0] >= 5
    assert positions[-1] <= 214


def test_map_rejects_tampered_q_and_reverse_motion() -> None:
    mapping = build_map(40, 40, [0.5, 1.0, 0.0], positions=[5, 17, 29])
    mapping["q"][10] += 0.25
    with pytest.raises(ProtocolError, match="saved knots"):
        validate_map(mapping)

    reversing = build_map(40, 40, [3.0, -3.0, 0.0], positions=[5, 17, 29])
    with pytest.raises(ProtocolError):
        validate_map(reversing)


def test_protected_tail_and_padded_frames_stay_fixed() -> None:
    mapping = build_map(40, 36, [0.5, 0.75, 0.0], positions=[5, 17, 29])
    delta = np.asarray(mapping["delta"])

    assert np.array_equal(delta[:5], np.zeros(5))
    assert np.array_equal(delta[31:], np.zeros(9))
    validate_map(mapping)


def test_nearest_uses_half_up_and_mirror_is_valid_for_small_map() -> None:
    positions = [5, 17, 29]
    mapping = build_map(40, 40, [0.5, 0.5, 0.0], positions=positions)
    source = np.zeros((40, 1, 1, 3), dtype=np.uint8)
    source[5, 0, 0, 0] = 12
    source[6, 0, 0, 0] = 99
    nearest = render_nearest(source, mapping)

    assert nearest[5, 0, 0, 0] == 99
    mirrored = mirror_map(mapping)
    validate_map(mirrored)


def test_regularization_uses_normalized_displacement_and_velocity_change() -> None:
    delta = np.asarray([0.0, 0.5, 0.5, 0.0], dtype=np.float64)
    expected = np.mean((delta / 3.0) ** 2) + np.mean((np.diff(delta) / 0.5) ** 2)

    assert regularization(delta) == pytest.approx(float(expected), abs=1e-12)


@pytest.mark.parametrize("amount", [-3, -2, -1, 1, 2, 3])
def test_global_seed_is_a_bounded_monotone_map(amount: int) -> None:
    mapping = global_seed(200, 198, amount)
    audit = validate_map(mapping)

    assert audit["max_abs_displacement"] <= 3.0
    assert audit["max_slope_change"] <= 0.5 + 1e-12
