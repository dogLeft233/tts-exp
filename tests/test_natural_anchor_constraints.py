from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_natural_anchor_constraints.py"
spec = importlib.util.spec_from_file_location("natural_anchor_constraints", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_norm_constraint_caps_frame_residuals() -> None:
    residual = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    constrained, stats = module.constrain_residual(residual, 0.5, None)
    assert torch.linalg.vector_norm(constrained, dim=-1).max().item() <= 5.0
    assert stats["norm_clipped_fraction"] > 0.0


def test_velocity_constraint_caps_increments() -> None:
    residual = torch.tensor([[0.0, 0.0], [1.0, 0.0], [11.0, 0.0]])
    constrained, stats = module.constrain_residual(residual, None, 0.5)
    velocity = torch.linalg.vector_norm(constrained[1:] - constrained[:-1], dim=-1)
    assert velocity.max().item() <= 5.5 + 1e-6
    assert stats["velocity_clipped_fraction"] > 0.0


def test_constraint_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="quantile"):
        module.constrain_residual(torch.zeros(3, 2), 0.0, None)
