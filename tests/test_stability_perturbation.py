"""Unit tests for scripts/25_stability_perturbation_syncnet.py.

These tests exercise the stability-targeted PGD transform, the random-noise
control transform, token-boundary loading, the post-hoc verification, and
the intervention registry. GPU/HuBERT-forward paths are mocked where needed.
"""

from __future__ import annotations

import importlib.util
import json
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
_S16 = _load_module("16_feature_separability.py")


def test_module_imports():
    assert hasattr(_S25, "stability_loss")
    assert hasattr(_S25, "load_token_boundaries")
    assert hasattr(_S25, "random_sign_noise_transform")
    assert hasattr(_S25, "pgd_stability_transform")
    assert hasattr(_S25, "_build_interventions")
    assert hasattr(_S25, "post_hoc_verify")
    assert hasattr(_S25, "main")


# ---------------------------------------------------------------------------
# stability_loss
# ---------------------------------------------------------------------------


def test_stability_loss_zero_for_identical_consecutive_frames():
    import torch

    h = torch.tensor([
        [[1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0]],
    ])
    frame_times = np.array([0.0, 0.02, 0.04, 0.06])
    boundary_idx = np.array([], dtype=np.int64)
    loss = _S25.stability_loss(h, frame_times, boundary_idx)
    assert float(loss) == pytest.approx(0.0, abs=1e-8)


def test_stability_loss_matches_reference_orthogonal_pair():
    import torch

    h = torch.tensor([
        [[1.0, 0.0],
         [0.0, 1.0]],
    ])
    frame_times = np.array([0.0, 0.02])
    boundary_idx = np.array([], dtype=np.int64)
    loss = _S25.stability_loss(h, frame_times, boundary_idx)
    assert float(loss) == pytest.approx(1.0, abs=1e-6)


def test_stability_loss_excludes_boundary_pairs():
    import torch

    h = torch.tensor([
        [[1.0, 0.0],
         [0.0, 1.0],
         [0.0, 1.0]],
    ])
    frame_times = np.array([0.0, 0.02, 0.04])
    boundary_idx = np.array([1], dtype=np.int64)
    loss = _S25.stability_loss(h, frame_times, boundary_idx)
    assert float(loss) == pytest.approx(1.0, abs=1e-6)


def test_stability_loss_matches_numpy_reference_for_real_input():
    import torch

    rng = np.random.default_rng(0)
    T, D = 20, 16
    h_np = rng.standard_normal((T, D)).astype(np.float32)
    frame_times = np.arange(T, dtype=np.float32) * 0.02

    after_idx = np.array([3, 7, 11], dtype=np.int64)
    boundary_times = frame_times[after_idx].tolist()
    _bs_ref, ss_ref = _S16.boundary_sharpness(frame_times, h_np, boundary_times)

    boundary_idx_np = after_idx - 1
    h_t = torch.from_numpy(h_np).unsqueeze(0)
    loss = _S25.stability_loss(h_t, frame_times, boundary_idx_np)
    assert float(loss) == pytest.approx(float(ss_ref), abs=1e-6)


def test_stability_loss_is_differentiable():
    import torch

    h = torch.tensor(
        [[[1.0, 0.0], [0.95, 0.05]]], requires_grad=True
    )
    frame_times = np.array([0.0, 0.02])
    boundary_idx = np.array([], dtype=np.int64)
    loss = _S25.stability_loss(h, frame_times, boundary_idx)
    loss.backward()
    assert h.grad is not None
    assert h.grad.shape == h.shape


# ---------------------------------------------------------------------------
# Token boundary loading
# ---------------------------------------------------------------------------


def test_uniform_fallback_boundaries_produces_50ms_segments():
    boundaries = _S25.apply_uniform_fallback_boundaries(4.35)
    assert boundaries[0] == pytest.approx(0.05, abs=1e-6)
    assert boundaries[-1] == pytest.approx(4.30, abs=1e-6)
    assert boundaries[1] - boundaries[0] == pytest.approx(0.05, abs=1e-6)
    assert (boundaries < 4.35).all()


def test_load_token_boundaries_happy_path(tmp_path):
    import json as _json

    manifest = {
        "samples": [
            {
                "sample_id": 1,
                "condition": "tts",
                "tokens": [
                    {"start_s": 0.0, "end_s": 0.20},
                    {"start_s": 0.20, "end_s": 0.45},
                    {"start_s": 0.45, "end_s": 0.70},
                ],
            },
            {
                "sample_id": 1,
                "condition": "natural",
                "tokens": [
                    {"start_s": 0.0, "end_s": 0.30},
                    {"start_s": 0.30, "end_s": 0.60},
                ],
            },
        ]
    }
    p = tmp_path / "alignment.json"
    p.write_text(_json.dumps(manifest))
    repo = tmp_path

    b_tts = _S25.load_token_boundaries(1, "tts", repo, manifest_path=p)
    b_nat = _S25.load_token_boundaries(1, "natural", repo, manifest_path=p)
    assert list(b_tts) == pytest.approx([0.20, 0.45], abs=1e-6)
    assert list(b_nat) == pytest.approx([0.30], abs=1e-6)


def test_load_token_boundaries_falls_back_when_sample_missing(tmp_path):
    import json as _json

    manifest = {"samples": [
        {"sample_id": 1, "condition": "tts", "tokens": [
            {"start_s": 0.0, "end_s": 0.30},
        ]}
    ]}
    p = tmp_path / "alignment.json"
    p.write_text(_json.dumps(manifest))

    boundaries = _S25.load_token_boundaries(
        2, "tts", tmp_path, manifest_path=p, audio_duration_s=1.0,
    )
    assert len(boundaries) > 0
    assert boundaries[0] == pytest.approx(0.05, abs=1e-6)
