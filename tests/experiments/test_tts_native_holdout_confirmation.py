import numpy as np

from scripts.experiments.tts_native_holdout_confirmation import native_policy
from scripts.experiments.tts_native_permutation import generate_perms, curve_metrics
from scripts.experiments.tts_native_boundary_audit import stats


def test_valid_policy_excludes_only_out_of_bounds_audio_pairs():
    L = 50
    matrix = np.ones((L, 31))
    for s in range(-15, 16):
        for i in range(L):
            if not 0 <= i+s < L:
                matrix[i, s+15] = 100
    assert native_policy(matrix, "valid")["C"] == 0
    assert native_policy(matrix, "guard20")["C"] == 0
    assert native_policy(matrix, "guard0")["C"] > 0


def test_permutation_seal_preserves_regions_and_interior_anchor_multiset():
    L, k = 67, 3
    domain = np.arange(L-k)
    region = np.full(L, -1, dtype=np.int32)
    region[domain] = np.where(domain < 20, 0, np.where(domain < L-20, 1, 2))
    pi, _ = generate_perms(L, domain, region, np.zeros(L, dtype=np.int32), "holdout|test|N|3", "whole", 256)
    assert pi.shape == (256, L)
    assert np.all(region[pi[:, domain]] == region[domain])
    assert np.all(np.sort(pi[:, 20:L-20], axis=1) == np.arange(20, L-20))
    again, _ = generate_perms(L, domain, region, np.zeros(L, dtype=np.int32), "holdout|test|N|3", "whole", 256)
    np.testing.assert_array_equal(pi, again)


def test_speaker_weighting_and_instance_scores_precede_pooling():
    assert stats([1., 1., 7.], ["s1", "s1", "s2"])["mean"] == 4.
    curves = np.ones((2, 31))
    curves[0, 10] = 0
    curves[1, 20] = 0
    separate = curve_metrics(curves, 3)[:, 0].mean()
    mixed = curve_metrics(curves.mean(0), 3)[0, 0]
    assert separate == 1 and mixed == .5
