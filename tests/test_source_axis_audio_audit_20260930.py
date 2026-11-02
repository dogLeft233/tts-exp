from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/source_axis_audio_audit_20260930.py"
SPEC = importlib.util.spec_from_file_location("source_axis_audio_audit_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_length_preservation_trims_only_the_single_rightmost_excess_sample():
    converted = np.array([10, 20, 30, 40], dtype=np.int16)
    preserved, trimmed = audit.preserve_roundtrip_frame_count(np, converted, 3)
    assert preserved.tolist() == [10, 20, 30]
    assert trimmed == 1


def test_length_preservation_keeps_same_length_pcm_exactly():
    converted = np.array([10, 20, 30], dtype=np.int16)
    preserved, trimmed = audit.preserve_roundtrip_frame_count(np, converted, 3)
    assert np.array_equal(preserved, converted)
    assert trimmed == 0


@pytest.mark.parametrize("roundtrip,reference_frames", [([1, 2], 3), ([1, 2, 3, 4, 5], 3), ([1], 0)])
def test_length_preservation_rejects_padding_or_more_than_one_sample_trim(roundtrip, reference_frames):
    with pytest.raises(ValueError):
        audit.preserve_roundtrip_frame_count(np, np.asarray(roundtrip), reference_frames)
