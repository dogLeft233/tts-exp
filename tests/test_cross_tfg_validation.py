"""Tests for cross-TFG mechanism result loading."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
_SCRIPT = _REPO / "scripts" / "21_validate_causal_results.py"
_SPEC = importlib.util.spec_from_file_location("_cross_tfg_validation", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["_cross_tfg_validation"] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_mechanism_name_accepts_feature_link_schema():
    """Phase 1 candidates use ``mechanism`` rather than ``name``."""
    assert _MODULE.mechanism_name({"mechanism": "segment_stability"}) == "segment_stability"


def test_aggregate_per_sample_deltas_produces_model_delta():
    """Per-sample model results must be usable by consistency analysis."""
    deltas = {
        "sync_c": {1: 1.0, 2: 3.0},
        "sync_d": {1: -2.0, 2: 0.0},
    }
    assert _MODULE.aggregate_per_sample_deltas(deltas) == {
        "delta_c": 2.0,
        "delta_d": -1.0,
    }
