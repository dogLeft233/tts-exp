import numpy as np

from scripts.experiments import mfa_linear_phone_gradient as pilot


def test_strength_grid_and_fixed_three_sample_scope():
    assert pilot.STRENGTHS == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert len(pilot.SIDS) == 3


def test_inverse_warp_zero_is_pcm_identity():
    values = np.arange(160, dtype=np.int16)
    anchors = np.array([0.0, 0.005, 0.0099375])
    assert np.array_equal(
        pilot.inverse_warp(values, anchors, anchors, 0.0, rate=16000.0), values
    )


def test_phone_alignment_does_not_pair_different_broad_classes():
    natural = [pilot.Interval(0, 0.1, "p"), pilot.Interval(0.1, 0.2, "a")]
    candidate = [pilot.Interval(0, 0.1, "k"), pilot.Interval(0.1, 0.2, "a")]
    assert pilot.align_word_phones(natural, candidate) == [(1, 1)]
