"""Tests for the English Gate 1 statistics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import json
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "30_analyze_english_wav2sem.py"
_spec = importlib.util.spec_from_file_location("_english_analysis", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)


def test_holm_adjust_orders_and_caps():
    adjusted = _mod.holm_adjust([0.01, 0.04, 0.20])
    assert adjusted == pytest.approx([0.03, 0.08, 0.20])


def test_exact_sign_flip_detects_consistent_direction():
    differences = np.ones(13)
    assert _mod.exact_sign_flip_pvalue(differences) == pytest.approx(2 / 8192)


def test_exact_sign_flip_is_one_for_zero_effect():
    assert _mod.exact_sign_flip_pvalue(np.zeros(5)) == 1.0


def test_summarize_reports_paired_effect():
    values = np.array([1.0, 2.0, 3.0])
    summary = _mod.summarize(values)
    assert summary["mean"] == 2.0
    assert summary["cohens_dz"] == 2.0
    assert len(summary["bootstrap_95_ci"]) == 2
