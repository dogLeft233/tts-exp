from scripts.experiments import mfa_linear_mechanism_pilot as pilot


def test_frozen_scope_is_three_records_and_fixed_arms():
    assert len(pilot.SIDS) == 3
    assert pilot.ARMS == ("N", "M", "M_REPAIR", "M_GL", "M_DIRECT_FIXED")


def test_metric_is_syncnet_curve_contrast():
    import numpy as np

    value = pilot.metrics(np.array([[4.0, 2.0, 4.0], [4.0, 2.0, 4.0]]), [0, 1])
    assert value == {"C": 2.0, "D": 2.0, "offset": 14}
