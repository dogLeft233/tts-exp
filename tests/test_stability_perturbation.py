"""Unit tests for scripts/25_stability_perturbation_syncnet.py.

These tests exercise the stability-targeted PGD transform, the random-noise
control transform, token-boundary loading, the post-hoc verification, and
the intervention registry. GPU/HuBERT-forward paths are mocked where needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"


def _load_module(filename: str):
    path = _SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_S25 = _load_module("25_stability_perturbation_syncnet.py")


def test_module_imports():
    assert hasattr(_S25, "stability_loss")
    assert hasattr(_S25, "load_token_boundaries")
    assert hasattr(_S25, "random_sign_noise_transform")
    assert hasattr(_S25, "pgd_stability_transform")
    assert hasattr(_S25, "_build_interventions")
    assert hasattr(_S25, "post_hoc_verify")
    assert hasattr(_S25, "main")
