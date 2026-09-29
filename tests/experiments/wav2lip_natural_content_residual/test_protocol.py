from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.wav2lip_natural_content_residual import config
from scripts.experiments.wav2lip_natural_content_residual.analysis import _pass_contrast, bootstrap_indices
from scripts.experiments.wav2lip_natural_content_residual.common import ProtocolError
from scripts.experiments.wav2lip_natural_content_residual.drivers import construct_arms


def _parent_arrays() -> dict[str, np.ndarray]:
    shape = (80, 12)
    zero = np.zeros(shape, dtype=np.float32)
    paired = zero.copy()
    wrong = zero.copy()
    paired[:, 3:6] = 0.2
    wrong[:, 3:6] = 0.1
    return {"PAIRED_TTS": paired, "SAME_PHONE_WRONG_INSTANCE": wrong, "NAT_ONLY": zero}


def test_zero_outside_k_and_deterministic_shuffle() -> None:
    natural = np.zeros((80, 12), dtype=np.float32)
    arms_a, meta_a = construct_arms("sample", natural, _parent_arrays(), [3, 4, 5])
    arms_b, meta_b = construct_arms("sample", natural, _parent_arrays(), [3, 4, 5])
    assert all(np.array_equal(arms_a[name], arms_b[name]) for name in config.ARMS)
    assert meta_a["shuffle_permutation"] == meta_b["shuffle_permutation"]
    outside = [index for index in range(12) if index not in {3, 4, 5}]
    for name in ("CORRECT", "WRONG", "SHUFFLE"):
        assert np.array_equal(arms_a[name][:, outside], natural[:, outside])
    assert meta_a["norm_control_valid"] is True


def test_degenerate_residual_blocks() -> None:
    natural = np.zeros((80, 12), dtype=np.float32)
    zero = np.zeros_like(natural)
    with pytest.raises(ProtocolError, match="INPUT_DEGENERATE"):
        construct_arms("sample", natural, {"PAIRED_TTS": zero, "SAME_PHONE_WRONG_INSTANCE": zero, "NAT_ONLY": zero}, [3, 4])


def test_shared_bootstrap_and_gate_require_all_three_metrics() -> None:
    labels, indices = bootstrap_indices(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f", "g", "g", "h", "h"])
    assert labels == sorted(labels)
    assert indices.shape == (config.BOOTSTRAP_DRAWS, config.EXPECTED_GROUP_COUNT)
    passing = {"metrics": {name: {"ci95": [0.01, 1.0]} for name in ("C", "D", "A")}, "group_joint_positive_count": 8}
    failing = {"metrics": {"C": {"ci95": [0.01, 1.0]}, "D": {"ci95": [-0.1, 1.0]}, "A": {"ci95": [0.01, 1.0]}}, "group_joint_positive_count": 8}
    assert _pass_contrast(passing)
    assert not _pass_contrast(failing)
