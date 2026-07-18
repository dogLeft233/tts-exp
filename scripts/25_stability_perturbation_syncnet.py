#!/usr/bin/env python3
"""25_stability_perturbation_syncnet.py — Stability-targeted adversarial perturbation.

Phase 2 stability-targeted intervention: surgically perturb HuBERT layer-11
``segment_stability`` on TTS and natural audio via PGD under a tight L_inf
budget, then re-run the Ditto + SyncNet pipeline used by script 22. A
random-direction noise control at the same epsilon provides a tighter
baseline than script 23's identity control.

Three intervention cells (n=9 AISHELL-1 paired samples, excl. sample 9):

  * ``stability_adj_tts``  — PGD raises the cost (destabilize TTS)        -> expect Sync-C down
  * ``stability_adj_nat``  — PGD lowers  the cost (stabilize natural)    -> expect Sync-C up
  * ``random_noise_tts``   — random ±eps sign-noise on TTS               -> control baseline

Post-hoc verification recomputes ``segment_stability`` on the perturbed
audio and the LUFS / spectral_tilt / dynamic-range drifts; cells whose
metric moves the wrong way or whose acoustic drift exceeds the documented
threshold are flagged in the per-sample JSON.

See ``docs/superpowers/specs/2026-07-18-stability-intervention-design.md``
for the full design.

Usage:
    python scripts/25_stability_perturbation_syncnet.py --run-id <RUN_ID> --dry-run
    python scripts/25_stability_perturbation_syncnet.py --run-id <RUN_ID> --samples 1
    python scripts/25_stability_perturbation_syncnet.py --run-id <RUN_ID> --eps 0.005 --alpha 0.001 --pgd-steps 50
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from tfg_feature_common import STUDY_SAMPLES, TARGET_SR, load_audio_mono, ensure_output_dirs


# ---------------------------------------------------------------------------
# Dynamic module loading (scripts start with digits so normal imports fail)
# ---------------------------------------------------------------------------


def _load_script_module(filename: str):
    """Load a sibling script as a module by file path (mirrors script 22)."""
    path = _REPO / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_s22 = _load_script_module("22_dose_response_syncnet.py")
_s15 = _load_script_module("15_extract_ssl_embeddings.py")
_s16 = _load_script_module("16_feature_separability.py")
_s20 = _load_script_module("20_causal_feature_interventions.py")

Intervention = _s22.Intervention
run_intervention_pipeline = _s22.run_intervention_pipeline
load_baseline_results = _s22.load_baseline_results
aggregate_deltas = _s22.aggregate_deltas

load_model = _s15.load_model
extract_frame_embeddings = _s15.extract_frame_embeddings
_compute_frame_stride = _s15._compute_frame_stride

boundary_sharpness = _s16.boundary_sharpness

compute_lufs = _s20.compute_lufs
compute_spectral_tilt = _s20.compute_spectral_tilt
compute_energy_env_std = _s20.compute_energy_env_std


# ---------------------------------------------------------------------------
# Locked-in symbol stubs (filled in by later tasks)
# ---------------------------------------------------------------------------


def stability_loss(h_l11, frame_times_np: np.ndarray, boundary_idx_np: np.ndarray):
    """Differentiable `segment_stability` cost matching script 16's metric.

    Parameters
    ----------
    h_l11 : torch.Tensor, shape (1, T, D) — hidden states at HuBERT layer 11
    frame_times_np : np.ndarray, shape (T,) — frame centre times in seconds
                      (only used for shape consistency; indices follow order)
    boundary_idx_np : np.ndarray, shape (B,) — frame *indices* that are
                       token-boundary frames (the "before" frame of each
                       boundary). Pairs (i, i+1) where i is in this set are
                       excluded from the within-segment average, mirroring
                       script 16.

    Returns
    -------
    loss : torch.Tensor scalar
        Mean (1 - cos_sim) over within-segment consecutive frame pairs.
    """
    import torch

    if h_l11.dim() != 3 or h_l11.shape[0] != 1:
        raise ValueError(
            f"expected h_l11 of shape (1, T, D), got {tuple(h_l11.shape)}"
        )
    T = h_l11.shape[1]
    if T < 2:
        raise ValueError("need >= 2 frames to compute stability")

    h = h_l11.squeeze(0)
    norm = h.norm(dim=1, keepdim=True).clamp_min(1e-10)
    h_norm = h / norm

    cos_sim = (h_norm[:-1] * h_norm[1:]).sum(dim=1)
    cost_pair = 1.0 - cos_sim

    if boundary_idx_np.size > 0:
        before_set = set(int(i) for i in boundary_idx_np)
        mask_vals = [0.0 if i in before_set else 1.0 for i in range(T - 1)]
        mask = torch.tensor(mask_vals, dtype=h.dtype, device=h.device)
    else:
        mask = torch.ones(T - 1, dtype=h.dtype, device=h.device)

    masked = cost_pair * mask
    if float(mask.sum().item()) < 1.0:
        return torch.zeros((), dtype=h.dtype, device=h.device)
    return masked.sum() / mask.sum()


def apply_uniform_fallback_boundaries(audio_duration_s: float) -> np.ndarray:
    raise NotImplementedError


def load_token_boundaries(sample_id: int, condition: str, repo: Path,
                          manifest_path: Path | None = None) -> np.ndarray:
    raise NotImplementedError


def random_sign_noise_transform(y_src, _y_other, _sr, _sid, eps: float = 0.005):
    raise NotImplementedError


def pgd_stability_transform(direction: str, eps: float = 0.005, alpha: float = 0.001,
                            K: int = 50, device: str = "cuda"):
    raise NotImplementedError


def pgd_perturb(y, model, frame_stride, dt_boundaries, eps, alpha, K, device, direction):
    raise NotImplementedError


def _build_interventions(args) -> list[Intervention]:
    raise NotImplementedError


def post_hoc_verify(y_pre, y_post, sr, sid, condition, repo, manifest_path=None):
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError
