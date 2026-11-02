import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('pilot_audit', Path(__file__).parents[1] /
    'scripts/experiments/nrt_two_arm_audit_20261023.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_raw_distance_confidence_uses_median_minus_best_not_mean():
    curve = np.full(31, 10.0)
    curve[12] = 4
    curve[0] = 100
    c, d, offset = m.raw_score(np.tile(curve, (7, 1)))
    assert (c, d, offset) == (6, 4, 3)
    with pytest.raises(ValueError):
        m.raw_score(np.zeros((2, 30)))
