from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_natural_anchor_boundary_silence.py"
spec = importlib.util.spec_from_file_location("natural_anchor_boundary_silence", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _tokens() -> list[dict[str, object]]:
    return [
        {"token": "sil", "start_s": 0.0, "end_s": 0.04, "is_silence": True},
        {"token": "a", "start_s": 0.04, "end_s": 0.12},
        {"token": "b", "start_s": 0.12, "end_s": 0.20},
        {"token": "sil", "start_s": 0.20, "end_s": 0.28, "is_silence": True},
    ]


def test_silence_mask_selects_only_silence_frames() -> None:
    mask = module.silence_frame_mask(_tokens(), 20)
    assert mask.any()
    assert not mask.all()
    assert mask[0]


def test_boundary_smoothing_changes_only_boundary_window() -> None:
    residual = torch.zeros(20, 2)
    residual[8:12] = 10.0
    output, stats = module.apply_condition(residual, _tokens(), "boundary_smooth_r2")
    assert stats["boundary_frame_fraction"] > 0.0
    assert torch.equal(output[:3], residual[:3])
    assert not torch.equal(output[8:12], residual[8:12])


def test_silence_anchor_zeroes_natural_silence() -> None:
    residual = torch.ones(20, 2)
    output, stats = module.apply_condition(residual, _tokens(), "silence_anchor")
    assert stats["silence_anchor"] is True
    assert torch.equal(output[module.silence_frame_mask(_tokens(), 20)], torch.zeros_like(output[module.silence_frame_mask(_tokens(), 20)]))
