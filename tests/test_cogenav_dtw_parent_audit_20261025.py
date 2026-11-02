"""Contracts for independent pilot audit."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('dtw_parent', Path(__file__).parents[1]/'scripts/experiments/cogenav_dtw_parent_audit_20261025.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pause_clock_has_no_reverse_match():
    segments = np.array([[0, 16000, 0, 16000], [16000, 32000, 21120, 37120]])
    truth, valid = module.clock(np.array([.5, 1.1, 1.82]), segments, reverse=True)
    assert valid.tolist() == [True, False, True]
    assert np.isnan(truth[1]) and abs(truth[2]-1.5) < 1e-12


def test_independent_optimum_and_path():
    cost = np.ones((4, 4))
    np.fill_diagonal(cost, 0)
    assert module.solve_cost(cost) == 0
    module.validate_path(np.column_stack([np.arange(4), np.arange(4)]), 4, 4)
