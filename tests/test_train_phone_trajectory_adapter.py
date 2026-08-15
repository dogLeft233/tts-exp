from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_phone_trajectory_adapter.py"
spec = importlib.util.spec_from_file_location("train_phone_trajectory_adapter", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_adapter_is_identity_initialized() -> None:
    mean = torch.zeros(module.FEATURE_DIM)
    std = torch.ones(module.FEATURE_DIM)
    adapter = module.PhoneTrajectoryAdapter(mean, std, hidden_channels=8, dilations=(1,))
    features = torch.randn(2, 12, module.FEATURE_DIM)
    assert torch.equal(adapter(features), features)


def test_adapter_rejects_wrong_feature_shape() -> None:
    adapter = module.PhoneTrajectoryAdapter(torch.zeros(module.FEATURE_DIM), torch.ones(module.FEATURE_DIM), hidden_channels=8, dilations=(1,))
    with pytest.raises(ValueError, match="features"):
        adapter(torch.zeros(1, 12, 8))


def test_corruption_preserves_shape_and_finite_values() -> None:
    features = torch.randn(20, module.FEATURE_DIM)
    tokens = [
        {"start_s": 0.0, "end_s": 0.10},
        {"start_s": 0.10, "end_s": 0.20},
        {"start_s": 0.20, "end_s": 0.30},
    ]
    corrupted = module.corrupt_trajectory(features, tokens, module.random.Random(0))
    assert corrupted.shape == features.shape
    assert torch.isfinite(corrupted).all()


def test_localized_corruption_changes_the_selected_crop() -> None:
    features = torch.sin(torch.arange(20 * module.FEATURE_DIM, dtype=torch.float32).reshape(20, module.FEATURE_DIM) / 17.0)
    tokens = [
        {"start_s": 0.0, "end_s": 0.10},
        {"start_s": 0.10, "end_s": 0.20},
        {"start_s": 0.20, "end_s": 0.30},
    ]
    local = module._localize_tokens(tokens, start_frame=2, frame_count=12)
    corrupted = module.corrupt_trajectory(features[2:14], local, module.random.Random(0))
    assert not torch.equal(corrupted, features[2:14])


    with pytest.raises(ValueError, match="features"):
        module.corrupt_trajectory(torch.zeros(5, 8), [], module.random.Random(0))
