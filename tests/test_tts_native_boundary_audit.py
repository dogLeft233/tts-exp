import numpy as np

from scripts.experiments.tts_native_boundary_audit import metric, shapley, summarize_matrix
from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine


def test_region_counts_and_closure():
    m = np.random.default_rng(2).uniform(2, 12, size=(71, 31))
    s = summarize_matrix(m)
    lag = np.arange(-15, 16)
    assert np.array_equal(s["regions"]["I"]["counts"], np.full(31, 31))
    assert np.array_equal(s["regions"]["E"]["counts"], 40-abs(lag))
    assert np.array_equal(s["regions"]["P"]["counts"], abs(lag))
    assert max(s["closure"].values()) < 1e-13
    assert np.allclose(s["metrics"]["guard20"]["curve"], s["regions"]["I"]["curve"])


def test_joint_length_rebuilds_audio_sentinel():
    v = np.full((44, 1024), .3, dtype=np.float32)
    a = np.full((44, 1024), .4, dtype=np.float32)
    old = SyncNetEngine.distance_matrix(v, a)
    new = SyncNetEngine.distance_matrix(v[:-1], a[:-1])
    assert old[42, 16] < new[42, 16]
    assert np.array_equal(old[:-1, :16], new[:, :16])


def test_nonlinear_shapley_closes_without_additive_region_c():
    rng = np.random.default_rng(42)
    n = summarize_matrix(rng.uniform(2, 10, (55, 31)))
    t = summarize_matrix(rng.uniform(3, 11, (91, 31)))
    for policy in ("guard0", "valid"):
        s = shapley(n, t, policy)
        assert s["closure"] < 1e-14
        for k in ("C", "B", "D"):
            assert abs(s[k]["total"]-(t["metrics"][policy][k]-n["metrics"][policy][k])) < 1e-13
    identity = shapley(n, n, "valid")
    assert identity["C"]["curve"] == identity["C"]["weight"] == 0


def test_guard_support_and_offset_sign():
    m = np.ones((40, 31))
    s = summarize_matrix(m)
    assert "guard20" not in s["metrics"]
    assert "guard15" in s["metrics"]
    z = np.ones(31)
    z[18] = 0
    assert metric(z)["offset"] == -3
    assert metric(z)["C"] == 1
