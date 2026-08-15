from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_natural_anchor_context_crossfade.py"
spec = importlib.util.spec_from_file_location("natural_anchor_context_crossfade", SCRIPT)
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


def test_crossfade_changes_only_boundary_context() -> None:
    residual = torch.zeros(20, 2)
    residual[6:] = 10.0
    output, stats = module.context_crossfade_residual(residual, _tokens(), radius_frames=2)
    assert stats["boundary_frame_fraction"] > 0.0
    assert torch.equal(output[:2], residual[:2])
    assert torch.equal(output[-3:], residual[-3:])
    assert not torch.equal(output[3:8], residual[3:8])


def test_crossfade_silence_anchor_zeroes_silence_frames() -> None:
    residual = torch.ones(20, 2)
    output, stats = module.context_crossfade_residual(residual, _tokens(), radius_frames=2, silence_anchor=True)
    mask = module._silence_mask(_tokens(), 20)
    assert stats["silence_anchor"] is True
    assert torch.equal(output[mask], torch.zeros_like(output[mask]))
