from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_aishell1_vocoder_domain_split.py"
spec = importlib.util.spec_from_file_location("aishell1_vocoder_domain_split", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_conditions_cover_feature_and_vocoder_axes() -> None:
    assert module.CONDITIONS == (
        "natural_direct_regular",
        "natural_direct_prematched",
        "raw_tts_direct_regular",
        "raw_tts_direct_prematched",
        "mfa_linear_regular",
        "mfa_linear_prematched",
    )


def test_exact_length_preserves_or_explicitly_adjusts() -> None:
    values = np.arange(5, dtype=np.float32)
    output, action = module.exact_length(values, 3)
    assert action == "right_crop"
    np.testing.assert_array_equal(output, np.array([0, 1, 2], dtype=np.float32))
    output, action = module.exact_length(values, 7)
    assert action == "right_zero_pad"
    np.testing.assert_array_equal(output, np.array([0, 1, 2, 3, 4, 0, 0], dtype=np.float32))


def test_exact_length_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="target_samples"):
        module.exact_length(np.ones(3, dtype=np.float32), 0)


def test_make_feature_condition_selects_expected_trajectory() -> None:
    natural = torch.ones(4, 3)
    tts = torch.full((5, 3), 2.0)
    mfa = torch.full((4, 3), 3.0)
    assert torch.equal(module.make_feature_condition("natural_direct", natural, tts, mfa), natural)
    assert torch.equal(module.make_feature_condition("raw_tts_direct", natural, tts, mfa), tts)
    assert torch.equal(module.make_feature_condition("mfa_linear", natural, tts, mfa), mfa)


def test_feature_distance_uses_common_prefix() -> None:
    left = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    right = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    result = module.feature_distance(left, right)
    assert result["compared_frames"] == 2
    assert result["cosine_distance"] == pytest.approx(0.5)


def test_audio_qc_tracks_native_length_and_finite_state() -> None:
    values = np.array([0.0, 0.5, -0.998, 1.0], dtype=np.float32)
    result = module._audio_qc(values, 16_000, 4)
    assert result["finite"] is True
    assert result["exact_natural_length"] is True
    assert result["clipped_sample_count"] == 1
