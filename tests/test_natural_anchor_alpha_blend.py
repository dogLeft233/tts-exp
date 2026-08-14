from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_natural_anchor_alpha_blend.py"
spec = importlib.util.spec_from_file_location("natural_anchor_alpha", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_blend_features_interpolates_between_anchors() -> None:
    natural = torch.zeros(3, 2)
    target = torch.full((3, 2), 4.0)
    assert torch.equal(module.blend_features(natural, target, 0.0), natural)
    assert torch.equal(module.blend_features(natural, target, 0.25), torch.ones(3, 2))
    assert torch.equal(module.blend_features(natural, target, 1.0), target)


def test_blend_features_rejects_invalid_alpha_and_shapes() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        module.blend_features(torch.zeros(2, 1), torch.zeros(2, 1), 1.1)
    with pytest.raises(ValueError, match="shape mismatch"):
        module.blend_features(torch.zeros(2, 1), torch.zeros(3, 1), 0.5)
