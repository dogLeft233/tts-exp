# Stability-Targeted Adversarial Perturbation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement script 25 (`scripts/25_stability_perturbation_syncnet.py`) that surgically perturbs HuBERT layer-11 `segment_stability` on TTS and natural audio via PGD, plus a same-ε random-noise control, then re-runs Ditto + SyncNet for 3 intervention cells × 9 samples. Result JSON feeds script 26 (separate, later).

**Architecture:** PGD under tight L_inf ε=0.005 budget, K=50 sign-step iterations, alpha=0.001. Random-direction noise shares the same ε. Module wiring reuses `Intervention` / `run_intervention_pipeline` / `load_baseline_results` / `aggregate_deltas` from script 22 and `load_model` / `extract_frame_embeddings` from script 15. Post-hoc verification recomputes `segment_stability` (sharing logic with `boundary_sharpness` from script 16) plus LUFS / spectral-tilt / dynamic-range drift for acoustic-confound detection. Cache-reuse conventions match script 22 so re-runs after cache hits are cheap.

**Tech Stack:** Python 3.11, PyTorch (gradient through HuBERT base), NumPy, librosa, soundfile, transformers (`AutoModel.from_pretrained`), pytest. Runs on remote AutoDL RTX 4080 conda env `ditto` + `syncnet`.

**Reference spec:** `docs/superpowers/specs/2026-07-18-stability-intervention-design.md`

---

## File Structure

**Create:**
- `scripts/25_stability_perturbation_syncnet.py` — main script
  - `stability_loss(h_l11, frame_times_np, boundary_idx_np)` — differentiable cost (Torch tensor in, Torch scalar out)
  - `load_token_boundaries(sample_id, condition, repo, manifest_path)` → `np.ndarray` of boundary end-times in seconds
  - `apply_uniform_fallback_boundaries(audio_duration_s)` → `np.ndarray` of 50ms-uniform end-times
  - `random_sign_noise_transform(y_src, eps)` → `np.ndarray` (matches Intervention's transform signature)
  - `pgd_stability_transform(direction, eps, alpha, K, device)` → closure `(y_src, _y_other, sr, sid) -> np.ndarray`
  - `pgd_perturb(y, model, frame_stride, dt_boundaries, eps, alpha, K, device, direction)` → numpy waveform
  - `_build_interventions(args)` → `list[Intervention]`
  - `post_hoc_verify(y_pre, y_post, sr, sid, condition, repo, manifest_path)` → dict
  - `main()` — CLI driver
- `tests/test_stability_perturbation.py` — unit tests

**Modify:**
- None. Script 25 imports from existing scripts; we don't touch them.

**Files imported from (read-only):**
- `scripts/tfg_feature_common.py`: `STUDY_SAMPLES`, `TARGET_SR`, `load_audio_mono`, `ensure_output_dirs`
- `scripts/15_extract_ssl_embeddings.py`: `load_model`, `extract_frame_embeddings`, `_compute_frame_stride`
- `scripts/16_feature_separability.py`: `boundary_sharpness` (returns `(boundary_change, segment_stability)`)
- `scripts/20_causal_feature_interventions.py`: `compute_lufs`, `compute_spectral_tilt`, `compute_energy_env_std`
- `scripts/22_dose_response_syncnet.py`: `Intervention`, `run_intervention_pipeline`, `load_baseline_results`, `aggregate_deltas`

---

## Task 1: Module skeleton + test file scaffold

**Files:**
- Create: `scripts/25_stability_perturbation_syncnet.py`
- Create: `tests/test_stability_perturbation.py`

- [ ] **Step 1: Create test file with a smoke import test**

`tests/test_stability_perturbation.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_stability_perturbation.py::test_module_imports -v
```

Expected: FAIL with module-not-found.

- [ ] **Step 3: Create empty script file with the required symbols**

`scripts/25_stability_perturbation_syncnet.py`:

```python
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


def stability_loss(h_l11, frame_times_np, boundary_idx_np):  # noqa: ANN001
    raise NotImplementedError


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
```

- [ ] **Step 4: Run import test to verify it passes**

```bash
pytest tests/test_stability_perturbation.py::test_module_imports -v
```

Expected: PASS (all stubs exist).

Note: `from scripts.X import _private` is allowed here — `_compute_frame_stride` is exposed by script 15 even though it starts with underscore. We keep it name-prefixed so it's obvious we're reaching across module boundaries for reuse, not adopting it as part of our public interface.

- [ ] **Step 5: Commit**

```bash
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "feat(script25): scaffold module + unit-test scaffold with stubs"
```

---

## Task 2: `stability_loss` (differentiable cost)

Mirrors the within-segment mean cosine-distance cost from `boundary_sharpness` in `scripts/16_feature_separability.py:369`, but operates on a torch tensor and uses frame *indices* for the boundary mask (so the gradient flows back through the model). The numeric result must match the numpy reference within 1e-6.

**Files:**
- Modify: `scripts/25_stability_perturbation_syncnet.py` (replace `stability_loss` stub)
- Test: `tests/test_stability_perturbation.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_stability_perturbation.py`:

```python
# ---------------------------------------------------------------------------
# stability_loss
# ---------------------------------------------------------------------------


def test_stability_loss_zero_for_identical_consecutive_frames():
    import torch

    # 4 frames, 3 dims. Identical consecutive frames -> cos_sim=1 -> 1-cos=0.
    h = torch.tensor([
        [[1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0]],
    ])
    frame_times = np.array([0.0, 0.02, 0.04, 0.06])
    boundary_idx = np.array([], dtype=np.int64)  # no boundaries -> all pairs in-segment
    loss = _S25.stability_loss(h, frame_times, boundary_idx)
    assert float(loss) == pytest.approx(0.0, abs=1e-8)


def test_stability_loss_matches_reference_orthogonal_pair():
    import torch

    # 2 frames orthogonal vectors -> cos_sim=0 -> 1-cos=1.
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

    # 3 frames: pairs (0,1) and (1,2). Boundary at frame 1 excludes pair (0,1),
    # leaving only (1,2). Frame 1 and 2 are orthogonal -> cost = 1.0.
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
    """End-to-end: the differentiable loss must match `boundary_sharpness`
    from script 16's segment_stability half on the same input."""
    import torch

    rng = np.random.default_rng(0)
    T, D = 20, 16
    h_np = rng.standard_normal((T, D)).astype(np.float32)
    frame_times = np.arange(T, dtype=np.float32) * 0.02
    boundary_idx_np = np.array([3, 7, 11], dtype=np.int64)

    # Reference: convert boundary indices -> boundary *times* and call
    # boundary_sharpness(frame_times, embeddings, token_boundaries).
    boundary_times = frame_times[boundary_idx_np].tolist()
    _bs_ref, ss_ref = _S16.boundary_sharpness(frame_times, h_np, boundary_times)

    # Differentiable version: pass (1, T, D) tensor + boundary indices.
    h_t = torch.from_numpy(h_np).unsqueeze(0)  # (1, T, D)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stability_perturbation.py -v -k stability_loss
```

Expected: 5 fails with `NotImplementedError`.

- [ ] **Step 3: Implement `stability_loss`**

Replace the `stability_loss` stub in `scripts/25_stability_perturbation_syncnet.py` with:

```python
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

    # L2-normalise per frame (matching script 16's emb_norm computation).
    h = h_l11.squeeze(0)  # (T, D)
    norm = h.norm(dim=1, keepdim=True).clamp_min(1e-10)
    h_norm = h / norm  # (T, D)

    # Cosine similarity between consecutive frames.
    cos_sim = (h_norm[:-1] * h_norm[1:]).sum(dim=1)  # (T-1,)
    cost_pair = 1.0 - cos_sim  # (T-1,)

    # Build a mask of within-segment pairs: pair index i corresponds to
    # frames (i, i+1) — exclude i if it's a "before-boundary" frame index.
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
```

Note: boundary indexes passed to `stability_loss` are the *before-boundary* frame indices (i.e. for a boundary at frame index k, the pair `(k-1, k)` is excluded). This is exactly the convention in `boundary_sharpness`'s `boundary_set.add(before_idx)` line.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stability_perturbation.py -v -k stability_loss
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "feat(script25): differentiable stability_loss matching script 16"
```

---

## Task 3: Token boundary loading + fallback

Load token boundaries from `data/wav2sem_analysis/manifest/alignment.json` (which lists per-sample tokens with `start_s` / `end_s`). Falls back to uniform 50ms segmentation if a sample/condition is missing.

**Files:**
- Modify: `scripts/25_stability_perturbation_syncnet.py` (replace stubs of `apply_uniform_fallback_boundaries` and `load_token_boundaries`)
- Test: `tests/test_stability_perturbation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# Token boundary loading
# ---------------------------------------------------------------------------


def test_uniform_fallback_boundaries_produces_50ms_segments():
    # 4.35s -> boundaries at 0.05, 0.10, ..., 4.30 (last boundary < duration).
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

    # Sample 2 missing: should fall back to uniform-50ms segmentation.
    # Need audio duration — caller passes the manifest path + repo. Use a
    # synthetic audio file via librosa... but to keep test light, we stub by
    # passing `audio_duration_s` via a side argument. Read function signature.
    boundaries = _S25.load_token_boundaries(
        2, "tts", tmp_path, manifest_path=p, audio_duration_s=1.0,
    )
    assert len(boundaries) > 0
    assert boundaries[0] == pytest.approx(0.05, abs=1e-6)
```

Note: the third test forces `load_token_boundaries` to accept an optional `audio_duration_s` kwarg for the fallback path. This is a convenience for the test; the production caller also supplies it.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stability_perturbation.py -v -k "boundary"
```

Expected: 3 fails.

- [ ] **Step 3: Implement `apply_uniform_fallback_boundaries` and `load_token_boundaries`**

Replace both stubs in `scripts/25_stability_perturbation_syncnet.py` with:

```python
FALLBACK_SEGMENT_SEC: float = 0.05


def apply_uniform_fallback_boundaries(audio_duration_s: float) -> np.ndarray:
    """Return uniform-50ms segment-end-times under audio_duration_s.

    Matches script 16's uniform-segmentation fallback (50 ms uniform segments
    used when MFA phoneme alignments are unavailable for a sample).
    """
    if audio_duration_s <= 0.0:
        return np.empty(0, dtype=np.float64)
    end = float(np.floor(audio_duration_s / FALLBACK_SEGMENT_SEC)) * FALLBACK_SEGMENT_SEC
    if end <= 0.0:
        return np.empty(0, dtype=np.float64)
    return np.arange(
        FALLBACK_SEGMENT_SEC, end + FALLBACK_SEGMENT_SEC, FALLBACK_SEGMENT_SEC, dtype=np.float64,
    )


def _load_manifest_samples(manifest_path: Path) -> dict[tuple[int, str], list[float]]:
    """Return {(sample_id, condition): [end_s, end_s, ...]} from manifest JSON."""
    if not manifest_path or not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text())
    samples = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(samples, list):
        return {}
    out: dict[tuple[int, str], list[float]] = {}
    for entry in samples:
        sid = entry.get("sample_id")
        cond = entry.get("condition")
        tokens = entry.get("tokens", [])
        if sid is None or cond is None or not tokens:
            continue
        # Token boundaries = each token's end_s, dropping the last token end
        # (which equals segment end). Matches script 16's input convention.
        ends = [float(t["end_s"]) for t in tokens if "end_s" in t]
        # Drop the final boundary so the last segment's end is the audio end.
        if len(ends) >= 2:
            out[(int(sid), str(cond))] = ends[:-1]
    return out


def load_token_boundaries(
    sample_id: int,
    condition: str,
    repo: Path,
    manifest_path: Path | None = None,
    audio_duration_s: float | None = None,
    sr: int = TARGET_SR,
) -> np.ndarray:
    """Return per-token boundary end-times (seconds) for (sample, condition).

    Looks up `manifest_path` if provided; otherwise falls back to
    `repo / "data" / "wav2sem_analysis" / "manifest" / "alignment.json"`.

    If no entry exists for this (sample_id, condition), returns uniform 50 ms
    segment boundaries — requires `audio_duration_s` to be provided in seconds
    (caller computes from the loaded audio waveform). Raises a clear error if
    fallback would be triggered but `audio_duration_s` is None.
    """
    if manifest_path is None:
        manifest_path = repo / "data" / "wav2sem_analysis" / "manifest" / "alignment.json"
    table = _load_manifest_samples(manifest_path)
    key = (int(sample_id), str(condition))
    if key in table:
        return np.asarray(table[key], dtype=np.float64)
    # Fallback path.
    if audio_duration_s is None:
        # Last-resort: caller didn't give us a duration — we have nothing to
        # segment. Return empty array and let downstream code skip stability
        # loss over zero within-segment pairs.
        return np.empty(0, dtype=np.float64)
    return apply_uniform_fallback_boundaries(audio_duration_s)
```

The `condition` argument should be either `"tts"` or `"natural"` — caller normalizes from the Intervention's `.source` field. `manifest_path` resolves to `data/wav2sem_analysis/manifest/alignment.json` (the existing per-sample token-manifest produced by script 14).

- [ ] **Step 4: Run boundary tests to verify they pass**

```bash
pytest tests/test_stability_perturbation.py -v -k "boundary"
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "feat(script25): load_token_boundaries + uniform-fallback path"
```

---

## Task 4: Random-direction noise transform

The control intervention. Border-case: must always satisfy `|delta| <= eps` and return audio in `[-1, 1]`.

**Files:**
- Modify: `scripts/25_stability_perturbation_syncnet.py` (replace `random_sign_noise_transform` stub)
- Test: `tests/test_stability_perturbation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# random_sign_noise_transform
# ---------------------------------------------------------------------------


def test_random_sign_noise_clamps_within_eps():
    rng = np.random.default_rng(42)
    y = rng.uniform(-1.0, 1.0, 16000).astype(np.float32)
    y_pulse = _S25.random_sign_noise_transform(y, None, 16000, 1, eps=0.005)
    delta = y_pulse - y
    assert np.abs(delta).max() <= 0.005 + 1e-8


def test_random_sign_noise_output_is_bounded_waveform():
    y = np.ones(8000, dtype=np.float32)  # already at upper bound
    y_pulse = _S25.random_sign_noise_transform(y, None, 16000, 1, eps=0.005)
    assert y_pulse.min() >= -1.0 - 1e-8
    assert y_pulse.max() <= 1.0 + 1e-8


def test_random_sign_noise_different_seed_each_call_gives_different_delta():
    # Same input, two calls: deltas should differ (randomness is real).
    y = np.zeros(4096, dtype=np.float32)
    a = _S25.random_sign_noise_transform(y, None, 16000, 1, eps=0.005)
    b = _S25.random_sign_noise_transform(y, None, 16000, 2, eps=0.005)  # diff sid seed
    # Probability of bit-identical outputs is essentially 0.
    assert not np.allclose(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stability_perturbation.py -v -k "random_sign_noise"
```

Expected: 3 fails.

- [ ] **Step 3: Implement `random_sign_noise_transform`**

Replace the stub with:

```python
def random_sign_noise_transform(
    y_src: np.ndarray,
    _y_other: np.ndarray | None,
    sr: int,
    sid: int,
    eps: float = 0.005,
) -> np.ndarray:
    """Random ±eps sign-pattern noise, scaled by uniform magnitude in [0, eps].

    Matches the L_inf budget of the PGD transforms. Returns float32 in
    `[-1, 1]`. The control intervention's purpose is to provide a
    tighter baseline than script 23's identity control: shared ε budget,
    non-targeted direction.

    Parameters
    ----------
    y_src : np.ndarray, shape (n_samples,)
        Source audio waveform (float32, in [-1, 1]).
    sid : int
        Sample id used to seed the RNG (deterministic per sample).
    eps : float
        L_inf bound on the perturbation.

    Returns
    -------
    np.ndarray, same shape/dtype as y_src
    """
    rng = np.random.default_rng(int(sid) * 12345 + 17)
    sign = rng.integers(0, 2, size=y_src.shape, dtype=np.int32) * 2 - 1  # ±1
    magnitude = rng.uniform(0.0, eps, size=y_src.shape).astype(np.float32)
    delta = (sign * magnitude).astype(np.float32)
    out = (y_src.astype(np.float32) + delta).clip(-1.0, 1.0)
    return out
```

The transform's signature `(y_src, y_other, sr, sid)` matches `Intervention.transform`'s `(y_tts, y_nat, sr, sid)` call convention from script 22 — the second arg is named `_y_other` because the random-noise control doesn't use it.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stability_perturbation.py -v -k "random_sign_noise"
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "feat(script25): random_sign_noise_transform control intervention"
```

---

## Task 5: PGD stability transform + L_inf projection

The actual adversarial loop: load HuBERT model once, gradient-step on the perturbation tensor, project to `|δ| ≤ eps`, clamp waveform to `[-1, 1]`. Direction encoding from the spec:
- `"raise_cost"` (cells with TTS source): descend `-L` -> `ascending L` -> less stable.
- `"lower_cost"` (cells with natural source): descend `L` -> lower cost -> more stable.

Test strategy: instead of running the full HuBERT base (huge real model), tests use a tiny stubbed differentiable model that returns `hidden_states[12]` built from a linear transform of the input waveform — the test only verifies projection + clamping + monotonic movement under the sign convention.

**Files:**
- Modify: `scripts/25_stability_perturbation_syncnet.py` (replace `pgd_stability_transform` and `pgd_perturb` stubs)
- Test: `tests/test_stability_perturbation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# pgd_perturb + pgd_stability_transform
# ---------------------------------------------------------------------------


class _StubHuBERT:
    """Tiny differentiable stand-in for HuBERT: produces (1, T, D)
    hidden states from a single linear transform of the raw waveform.

    For tests only — never used in production paths.
    """

    def __init__(self, T_out: int = 8, D: int = 4, seed: int = 0):
        import torch

        self.T_out = T_out
        self.D = D
        g = torch.Generator().manual_seed(seed)
        # Weight (D, 1) plus bias — output frame i depends on waveform sample i
        # (with broadcasting); a single conv-like view sums a window of 4 samples.
        self.weight = torch.randn(D, 1, generator=g) * 0.1
        self.bias = torch.randn(D, generator=g) * 0.05
        self.config = self._StubConfig()

    class _StubConfig:
        conv_stride = [5, 2, 2, 2, 2, 2, 2]
        num_hidden_layers = 13

    def __call__(self, waveform, output_hidden_states=True):
        import torch

        # waveform: (1, n) -> output 13 layers, each (1, T, D).
        # Only layer 12 (index 12) is nontrivial; rest are zeros.
        n = waveform.shape[-1]
        # Take every (n // T_out)-th window of 4 samples.
        step = max(1, n // self.T_out)
        frames = waveform[..., :step * self.T_out].reshape(1, self.T_out, step)
        # Linear map: h = weight @ frames (collapsed over step axis) + bias.
        # Use mean over step axis so gradient flows through frames.
        frames_mean = frames.mean(dim=2, keepdim=False).unsqueeze(-1)  # (1, T, 1)
        h_l12 = (frames_mean * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias  # (1, T, D)
        zeros_layer = torch.zeros_like(h_l12)
        hidden_states = tuple(zeros_layer for _ in range(12)) + (h_l12,)
        out = MagicMock()
        out.hidden_states = hidden_states
        return out


def _random_source(n: int = 8000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.uniform(-0.5, 0.5, n) * 0.5).astype(np.float32)


def test_pgd_perturb_delta_is_bounded_by_eps():
    import torch

    model = _StubHuBERT()
    y = _random_source(8000)
    dt_boundaries = np.array([0.02, 0.04, 0.06], dtype=np.float32)
    out = _S25.pgd_perturb(
        y, model, 640, dt_boundaries,
        eps=0.005, alpha=0.001, K=20, device="cpu", direction="raise_cost",
    )
    delta = out.astype(np.float64) - y.astype(np.float64)
    assert np.abs(delta).max() <= 0.005 + 1e-6


def test_pgd_perturb_output_is_clamped_to_valid_waveform():
    model = _StubHuBERT()
    y = np.ones(8000, dtype=np.float32)
    dt_boundaries = np.array([0.02, 0.04, 0.06], dtype=np.float32)
    out = _S25.pgd_perturb(
        y, model, 640, dt_boundaries,
        eps=0.005, alpha=0.001, K=20, device="cpu", direction="raise_cost",
    )
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_pgd_perturb_does_not_change_input_when_eps_is_zero():
    model = _StubHuBERT()
    y = _random_source(8000)
    dt_boundaries = np.array([0.02, 0.04, 0.06], dtype=np.float32)
    out = _S25.pgd_perturb(
        y, model, 640, dt_boundaries,
        eps=0.0, alpha=0.0, K=20, device="cpu", direction="raise_cost",
    )
    assert np.allclose(out, y, atol=1e-6)


def test_pgd_perturb_moves_metric_in_expected_direction():
    """For the stub model, raise_cost should *raise* the loss value."""
    import torch

    model = _StubHuBERT()
    y = _random_source(8000)
    dt_boundaries = np.array([0.02, 0.04, 0.06], dtype=np.float32)
    frame_stride = 640
    frame_times = np.arange(
        y.shape[0] // frame_stride, dtype=np.float32,
    ) * (frame_stride / _S25.TARGET_SR)
    # Boundary frame indices: 1, 2, 3 (after the 3 boundaries).
    boundary_idx = np.searchsorted(frame_times, dt_boundaries).astype(np.int64) - 1

    def compute_loss(audio_np):
        with torch.no_grad():
            x = torch.from_numpy(audio_np).float().unsqueeze(0)
            out = model(x)
            h_l11 = out.hidden_states[12]
            return float(_S25.stability_loss(h_l11, frame_times, boundary_idx))

    loss_pre = compute_loss(y)

    # raise_cost should increase the loss (lose stability).
    out_raise = _S25.pgd_perturb(
        y, model, frame_stride, dt_boundaries,
        eps=0.05, alpha=0.01, K=30, device="cpu", direction="raise_cost",
    )
    loss_post_raise = compute_loss(out_raise)
    assert loss_post_raise > loss_pre, "raise_cost should increase stability_loss"

    # lower_cost should decrease the loss.
    out_lower = _S25.pgd_perturb(
        y, model, frame_stride, dt_boundaries,
        eps=0.05, alpha=0.01, K=30, device="cpu", direction="lower_cost",
    )
    loss_post_lower = compute_loss(out_lower)
    assert loss_post_lower < loss_pre, "lower_cost should decrease stability_loss"


def test_pgd_stability_transform_creates_callable_with_signature_intervention_expects():
    """The Intervention.run_intervention_pipeline calls transform(y_tts, y_nat, sr, sid).
    The returned closure must accept this signature and return a numpy array.
    """
    impl = _S25.pgd_stability_transform(
        direction="raise_cost", eps=0.005, alpha=0.001, K=5, device="cpu",
    )
    # Provide a stub model via patch on _S25._ensure_hubert_model.
    orig_loader = _S25.__dict__.get("_ensure_hubert_model")
    _S25._ensure_hubert_model = lambda device: _StubHuBERT()  # type: ignore[attr-defined]
    try:
        y_tts = _random_source(4000)
        y_nat = _random_source(4000)
        out = impl(y_tts, y_nat, 16000, 1)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == y_tts.shape
        delta = np.abs(out.astype(np.float64) - y_tts.astype(np.float64)).max()
        assert delta <= 0.005 + 1e-6
    finally:
        if orig_loader is not None:
            _S25._ensure_hubert_model = orig_loader  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stability_perturbation.py -v -k "pgd"
```

Expected: 5 fails.

- [ ] **Step 3: Implement `pgd_perturb` and `_ensure_hubert_model`**

Add the loader (above the existing stubs in the script body):

```python
_HUBERT_CACHE: dict[str, object] = {}


def _ensure_hubert_model(device: str = "cuda"):
    """Return a cached HuBERT base model with output_hidden_states=True.

    The model is loaded once per process and reused across the per-sample
    PGD loop. Weights are frozen (eval mode, requires_grad=False on all
    parameters) — only the *input waveform* is a parameter.
    """
    if device in _HUBERT_CACHE:
        return _HUBERT_CACHE[device]
    import torch  # noqa: F401  (delayed import keeps CPU-only tests importing torch lazily)

    model_name = "facebook/hubert-base-ls960"
    model, embedding_dim, frame_stride, num_layers = load_model(
        model_name,
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _HUBERT_CACHE[device] = {
        "model": model,
        "frame_stride": int(frame_stride),
        "embedding_dim": int(embedding_dim),
        "num_layers": int(num_layers),
    }
    return _HUBERT_CACHE[device]
```

- [ ] **Step 4: Implement `pgd_perturb` (replacing the stub)**

```python
def pgd_perturb(
    y: np.ndarray,
    model,  # HuBERT model
    frame_stride: int,
    dt_boundaries: np.ndarray,
    eps: float,
    alpha: float,
    K: int,
    device: str,
    direction: str,
) -> np.ndarray:
    """Apply PGD to perturb raw waveform `y` toward a stability target.

    Parameters
    ----------
    y : np.ndarray, shape (n_samples,) float32
        Source waveform (values in [-1, 1]).
    model : HuBERT
        Frozen HuBERT model with output_hidden_states=True.
    frame_stride : int
        Audio samples per HuBERT output frame.
    dt_boundaries : np.ndarray
        Token-boundary *times* in seconds. Converted to frame indices inside.
    eps : float
        L_inf budget on the perturbation delta.
    alpha : float
        Per-step PGD step size.
    K : int
        Number of PGD iterations.
    device : str
        CPU or CUDA device string.
    direction : str
        "raise_cost" to increase stability_loss (less stable); "lower_cost"
        to decrease stability_loss (more stable).

    Returns
    -------
    np.ndarray, shape (n_samples,) float32 — perturbed waveform in [-1, 1].
    """
    import torch

    if direction not in ("raise_cost", "lower_cost"):
        raise ValueError(f"direction must be raise_cost|lower_cost, got {direction!r}")
    # Sign factor: combined with delta -= alpha * sign(delta.grad) below, this
    # gives descent on `direction_factor * L`.
    direction_factor = +1.0 if direction == "raise_cost" else -1.0

    y_t = torch.from_numpy(np.ascontiguousarray(y)).float().to(device)
    delta = torch.zeros_like(y_t, requires_grad=True)

    # Compute frame times once (constant across iterations).
    n_samples = y_t.shape[0]
    # Match extract_frame_embeddings: n_frames = hidden_states[0].shape[1].
    # Use a probe forward pass to get the actual frame count.
    with torch.no_grad():
        probe = model(y_t.unsqueeze(0), output_hidden_states=True)
        n_frames = probe.hidden_states[0].shape[1]
    frame_times = np.arange(n_frames, dtype=np.float32) * (
        float(frame_stride) / float(TARGET_SR)
    )

    # Convert boundary *times* -> before-boundary frame indices.
    if dt_boundaries.size > 0:
        before_idx = np.searchsorted(frame_times, dt_boundaries.astype(np.float32)) - 1
        before_idx = before_idx[(before_idx >= 0) & (before_idx < n_frames - 1)]
        boundary_idx_np = before_idx.astype(np.int64)
    else:
        boundary_idx_np = np.empty(0, dtype=np.int64)

    for _ in range(K):
        x_adv = (y_t + delta).clamp(-1.0, 1.0)
        outputs = model(x_adv.unsqueeze(0), output_hidden_states=True)
        h_l11 = outputs.hidden_states[12]
        L = stability_loss(h_l11, frame_times, boundary_idx_np)
        loss = direction_factor * L
        loss.backward()
        with torch.no_grad():
            grad_sign = torch.sign(delta.grad)
            # Descend on `loss` wrt delta:
            #   raise_cost -> descend -L -> delta += alpha * sign(grad)
            #   lower_cost -> descend  L -> delta -= alpha * sign(grad)
            # The unified formula: delta -= direction_factor * alpha * sign(grad)
            delta -= direction_factor * alpha * grad_sign
            # L_inf projection.
            delta.clamp_(-eps, eps)
        delta.grad = None

    with torch.no_grad():
        perturbed = (y_t + delta).clamp(-1.0, 1.0).squeeze()
    return perturbed.cpu().numpy().astype(np.float32)
```

Wait — double-check the sign convention:

- PGD descent step is `delta -= lr * sign(dL/d(delta))`. Here `loss = direction_factor * L`.
- For `raise_cost`, `direction_factor = +1`. We want `delta` that *increases* `L`. Since `delta` is the variable being optimized and we want to *maximize* L, we want *ascent on L*; equivalently descent on `-L` = `direction_factor = -1`. So `raise_cost -> direction_factor = -1`.

Let me redo:

- raise_cost: maximize L -> descend `-L` -> direction_factor = **-1**
- lower_cost: minimize L -> descend `L` -> direction_factor = **+1**

Fix the implementation:

```python
def pgd_perturb(
    y: np.ndarray,
    model,
    frame_stride: int,
    dt_boundaries: np.ndarray,
    eps: float,
    alpha: float,
    K: int,
    device: str,
    direction: str,
) -> np.ndarray:
    """Apply PGD to perturb raw waveform `y` toward a stability target.

    direction: "raise_cost" -> ascend L (less stable);
               "lower_cost" -> descend L (more stable).
    """
    import torch

    if direction not in ("raise_cost", "lower_cost"):
        raise ValueError(f"direction must be raise_cost|lower_cost, got {direction!r}")
    # We always *descend* on (direction_factor * L). raise_cost wants ascent
    # on L -> descend on -L -> direction_factor = -1.
    # lower_cost wants descent on L -> direction_factor = +1.
    direction_factor = -1.0 if direction == "raise_cost" else +1.0

    y_t = torch.from_numpy(np.ascontiguousarray(y)).float().to(device)
    delta = torch.zeros_like(y_t, requires_grad=True)

    n_samples = y_t.shape[0]
    with torch.no_grad():
        probe = model(y_t.unsqueeze(0), output_hidden_states=True)
        n_frames = probe.hidden_states[0].shape[1]
    frame_times = np.arange(n_frames, dtype=np.float32) * (
        float(frame_stride) / float(TARGET_SR)
    )

    if dt_boundaries.size > 0:
        before_idx = np.searchsorted(frame_times, dt_boundaries.astype(np.float32)) - 1
        before_idx = before_idx[(before_idx >= 0) & (before_idx < n_frames - 1)]
        boundary_idx_np = before_idx.astype(np.int64)
    else:
        boundary_idx_np = np.empty(0, dtype=np.int64)

    for _ in range(K):
        x_adv = (y_t + delta).clamp(-1.0, 1.0)
        outputs = model(x_adv.unsqueeze(0), output_hidden_states=True)
        h_l11 = outputs.hidden_states[12]
        L = stability_loss(h_l11, frame_times, boundary_idx_np)
        loss = direction_factor * L
        loss.backward()
        with torch.no_grad():
            grad_sign = torch.sign(delta.grad)
            delta -= direction_factor * alpha * grad_sign
            delta.clamp_(-eps, eps)
        delta.grad = None

    with torch.no_grad():
        perturbed = (y_t + delta).clamp(-1.0, 1.0).squeeze()
    return perturbed.cpu().numpy().astype(np.float32)
```

Wait, let me check the test:

```python
# raise_cost should increase the loss (lose stability).
out_raise = _S25.pgd_perturb(...)
loss_post_raise = compute_loss(out_raise)
assert loss_post_raise > loss_pre, "raise_cost should increase stability_loss"

# lower_cost should decrease the loss.
out_lower = _S25.pgd_perturb(...)
loss_post_lower = compute_loss(out_lower)
assert loss_post_lower < loss_pre, "lower_cost should decrease stability_loss"
```

With my fixed convention:
- raise_cost: `direction_factor = -1`, `loss = -L`, `delta += alpha * sign(grad)`. Since `grad = d(L)/d(delta)`, `delta` moves in direction of increasing L. So L goes up. ✓
- lower_cost: `direction_factor = +1`, `loss = +L`, `delta -= alpha * sign(grad)`. delta moves against increasing L. L goes down. ✓

The implementation is correct. Replace the stub in the file.

- [ ] **Step 5: Implement `pgd_stability_transform` (replacing the stub)**

```python
def pgd_stability_transform(
    direction: str,
    eps: float = 0.005,
    alpha: float = 0.001,
    K: int = 50,
    device: str = "cuda",
):
    """Return an Intervention-compatible transform closure.

    The Intervention contract from script 22 is:
        transform(y_tts, y_nat, sr, sid) -> np.ndarray (mono waveform)

    Which waveform goes in depends on the Intervention's `.source` field
    ("tts" -> y_tts; "natural" -> y_nat). The dispatch happens inside
    `run_intervention_pipeline`; this closure is called with whichever source
    was selected, paired with the *unused* counterpart (passed as `y_other`).

    We load HuBERT lazily on first call so dry-runs (which skip pipeline
    invocation) don't trigger GPU init.
    """
    def _impl(y_src, _y_other, sr, sid):
        if sr != TARGET_SR:
            raise ValueError(
                f"PGD transform requires sr={TARGET_SR}; got sr={sr}"
            )
        cache = _ensure_hubert_model(device)
        model = cache["model"]
        frame_stride = cache["frame_stride"]
        # We DON'T have token boundaries directly here; the per-sample dispatch
        # calls a wrapper `pgd_stability_intervention` which looks up
        # boundaries from the manifest given sid + condition; see below.
        # This transform is intended to be wrapped by `_build_interventions`
        # with sid-aware closure.
        raise NotImplementedError(
            "pgd_stability_transform requires sid-aware dispatch; "
            "use _build_interventions() to wire it"
        )
    return _impl
```

Hmm — the Intervention transform's signature is `(y_tts, y_nat, sr, sid)` so `sid` is passed. But we also need the boundary lookup per sid. The cleanest pattern is to have `_build_interventions()` create closures that close over the manifest and call `pgd_perturb` directly with the right boundaries for `sid`. Let's replace the simple `pgd_stability_transform` stub with a richer version that accepts `(repo, manifest_path, condition)`:

```python
def pgd_stability_transform(
    direction: str,
    eps: float = 0.005,
    alpha: float = 0.001,
    K: int = 50,
    device: str = "cuda",
    repo: Path | None = None,
    manifest_path: Path | None = None,
    condition: str = "tts",
    sr: int = TARGET_SR,
):
    """Return an Intervention-compatible transform closure.

    closure signature (matches script 22's Intervention.transform):
       _impl(y_tts, y_nat, sr, sid) -> np.ndarray (mono waveform)

    The closure picks `y_src` based on `condition` ("tts" -> y_tts,
    "natural" -> y_nat), computes audio_duration_s for fallback boundary
    loading, and dispatches to `pgd_perturb`.
    """
    if repo is None:
        repo = _REPO
    if manifest_path is None:
        manifest_path = repo / "data" / "wav2sem_analysis" / "manifest" / "alignment.json"
    if condition not in ("tts", "natural"):
        raise ValueError(f"condition must be tts|natural, got {condition!r}")

    def _impl(y_tts, y_nat, sr_arg, sid):
        if sr_arg != TARGET_SR:
            raise ValueError(
                f"PGD transform requires sr={TARGET_SR}; got sr={sr_arg}"
            )
        if condition == "tts":
            y_src = y_tts
        else:
            y_src = y_nat

        cache = _ensure_hubert_model(device)
        model = cache["model"]
        frame_stride = cache["frame_stride"]
        audio_duration_s = float(y_src.shape[0]) / float(TARGET_SR)
        boundaries = load_token_boundaries(
            sid, condition, repo, manifest_path=manifest_path,
            audio_duration_s=audio_duration_s,
        )
        return pgd_perturb(
            y_src, model, frame_stride, boundaries,
            eps=eps, alpha=alpha, K=K, device=device, direction=direction,
        )

    return _impl
```

- [ ] **Step 6: Run the PGD tests**

Update the test that previously patched `_S25._ensure_hubert_model` to use the new signature:

In `tests/test_stability_perturbation.py` modify `test_pgd_stability_transform_creates_callable_with_signature_intervention_expects`:

```python
def test_pgd_stability_transform_creates_callable_with_signature_intervention_expects():
    impl = _S25.pgd_stability_transform(
        direction="raise_cost", eps=0.005, alpha=0.001, K=5, device="cpu",
    )
    orig_loader = _S25._ensure_hubert_model
    _S25._ensure_hubert_model = lambda device: {
        "model": _StubHuBERT(),
        "frame_stride": 640,
        "embedding_dim": 4,
        "num_layers": 13,
    }
    try:
        y_tts = _random_source(4000)
        y_nat = _random_source(4000)
        out = impl(y_tts, y_nat, 16000, 1)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == y_tts.shape
        delta = np.abs(out.astype(np.float64) - y_tts.astype(np.float64)).max()
        assert delta <= 0.005 + 1e-6
    finally:
        _S25._ensure_hubert_model = orig_loader
```

(We rewrite the test rather than appending the new version during Step 1's stub, so this Step 6 reflects the pilot revision. If the test file already had the buggy loader patch code from Step 1's version, this is the rewrite.)

- [ ] **Step 7: Run all PGD tests to verify pass**

```bash
pytest tests/test_stability_perturbation.py -v -k "pgd"
```

Expected: 5 PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "feat(script25): PGD stability perturbation with L_inf projection"
```

---

## Task 6: Intervention registry `_build_interventions`

Wire 3 cells together. Returns a list of `Intervention` instances the main loop will iterate.

**Files:**
- Modify: `scripts/25_stability_perturbation_syncnet.py` (replace `_build_interventions` stub)
- Test: `tests/test_stability_perturbation.py`

- [ ] **Step 1: Write failing test**

Append:

```python
# ---------------------------------------------------------------------------
# _build_interventions
# ---------------------------------------------------------------------------


def test_build_interventions_returns_three_cells(monkeypatch):
    # Stub the PGD transform so we don't need HuBERT for construction.
    sentinel = object()
    monkeypatch.setattr(
        _S25, "pgd_stability_transform",
        lambda **kw: sentinel,
    )
    ns = argparse.Namespace(
        eps=0.005, alpha=0.001, pgd_steps=50, pgd_restarts=1,
        device="cpu", repo=_REPO,
        manifest=None,
    )
    cells = _S25._build_interventions(ns)
    assert len(cells) == 3
    names = [c.name for c in cells]
    assert names == [
        "stability_adj_tts",
        "stability_adj_nat",
        "random_noise_tts",
    ]
    assert all(hasattr(c, "transform") for c in cells)
    # Each cell's source + baseline_cond match the design spec.
    assert cells[0].source == "tts"
    assert cells[0].baseline_cond == "tts_raw"
    assert cells[1].source == "natural"
    assert cells[1].baseline_cond == "natural_raw"
    assert cells[2].source == "tts"
    assert cells[2].baseline_cond == "tts_raw"


def test_build_interventions_random_noise_uses_tts_source(monkeypatch):
    monkeypatch.setattr(
        _S25, "pgd_stability_transform",
        lambda **kw: (lambda y_t, y_n, sr, sid: y_t * 2),  # arbitrary
    )
    ns = argparse.Namespace(
        eps=0.005, alpha=0.001, pgd_steps=50, pgd_restarts=1,
        device="cpu", repo=_REPO,
        manifest=None,
    )
    cells = _S25._build_interventions(ns)
    # random_noise_tts transform should echo y_tts (since source=tts).
    # We patch random_sign_noise_transform to make this testable.
    sentinel_stubs = {"called": 0}
    def fake(y_src, _y_other, _sr, _sid, eps=0.005):
        sentinel_stubs["called"] += 1
        return y_src + 0.001  # tiny change, deterministic for assertion
    monkeypatch.setattr(_S25, "random_sign_noise_transform", fake)
    # Rebuild cells to pick up the patched function.
    cells = _S25._build_interventions(ns)
    y_t = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    y_n = np.array([-1.0, 1.0, -0.5], dtype=np.float32)
    out = cells[2].transform(y_t, y_n, 16000, 5)
    assert sentinel_stubs["called"] == 1
    assert out.shape == y_t.shape
    # The fake returned y_src+0.001 (i.e. modified y_tts, not y_nat).
    np.testing.assert_allclose(out, y_t + 0.001, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stability_perturbation.py -v -k "build_interventions"
```

Expected: 2 fails.

- [ ] **Step 3: Implement `_build_interventions`**

Replace the stub:

```python
def _build_interventions(args: argparse.Namespace) -> list[Intervention]:
    """Construct the 3 intervention cells defined by the spec."""
    cells: list[Intervention] = []
    # PGD cells use sid-aware closures (load manifest per-sample).
    manifest_path = args.manifest
    cells.append(Intervention(
        name="stability_adj_tts",
        source="tts",
        baseline_cond="tts_raw",
        transform_description=(
            "PGD raises HuBERT-L11 segment_stability cost on TTS (destabilize)"
        ),
        expected_sync_direction="decrease",
        transform=pgd_stability_transform(
            direction="raise_cost",
            eps=args.eps, alpha=args.alpha, K=args.pgd_steps,
            device=args.device, repo=args.repo, manifest_path=manifest_path,
            condition="tts",
        ),
    ))
    cells.append(Intervention(
        name="stability_adj_nat",
        source="natural",
        baseline_cond="natural_raw",
        transform_description=(
            "PGD lowers HuBERT-L11 segment_stability cost on natural (stabilize)"
        ),
        expected_sync_direction="increase",
        transform=pgd_stability_transform(
            direction="lower_cost",
            eps=args.eps, alpha=args.alpha, K=args.pgd_steps,
            device=args.device, repo=args.repo, manifest_path=manifest_path,
            condition="natural",
        ),
    ))
    # Random-noise control: closure wrapper around random_sign_noise_transform.
    eps_capture = args.eps
    def _random_noise(y_tts, _y_nat, sr, sid, eps=eps_capture):
        return random_sign_noise_transform(y_tts, None, sr, sid, eps=eps)
    cells.append(Intervention(
        name="random_noise_tts",
        source="tts",
        baseline_cond="tts_raw",
        transform_description=(
            f"Random ±eps uniform noise on TTS (eps={eps_capture}, control)"
        ),
        expected_sync_direction="decrease",  # same eps noise -> drops like other TTS cells
        transform=_random_noise,
    ))
    return cells
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stability_perturbation.py -v -k "build_interventions"
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "feat(script25): _build_interventions wiring for 3 cells"
```

---

## Task 7: Post-hoc verification

After DNA perturbation, re-extract HuBERT layer-11 embeddings on both pre and post audio using `extract_frame_embeddings`, recompute `segment_stability` (via `boundary_sharpness`), and measure acoustic drifts (LUFS / spectral tilt / dynamic-range std).

**Files:**
- Modify: `scripts/25_stability_perturbation_syncnet.py` (replace `post_hoc_verify` stub)
- Test: `tests/test_stability_perturbation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# post_hoc_verify
# ---------------------------------------------------------------------------


def test_post_hoc_verify_fields_present():
    """post_hoc_verify should return a dict with the expected fields, even
    on near-trivial input."""
    rng = np.random.default_rng(0)
    sr = _S25.TARGET_SR
    y_pre = (rng.uniform(-0.3, 0.3, 16000 * 2) * 0.5).astype(np.float32)
    y_post = (y_pre + rng.uniform(-0.005, 0.005, y_pre.shape)).astype(np.float32)
    # Monkey-patch extract_frame_embeddings + boundary_sharpness so we don't
    # need the real HuBERT model.
    import torch

    fake_h = torch.zeros((1, 50, 4), dtype=torch.float32)
    fake_times = np.arange(50, dtype=np.float32) * 0.02
    monkey_extract = lambda model, audio, sample_rate, layers, device="cpu": (
        fake_h.numpy(), fake_times
    )
    monkey_bs = (
        lambda frame_times, frame_emb, token_boundaries: (0.0, 0.1)
    )
    orig_extract = _S25.extract_frame_embeddings
    orig_bs = _S25.boundary_sharpness
    orig_load_model = _S25._ensure_hubert_model
    _S25.extract_frame_embeddings = monkey_extract
    _S25.boundary_sharpness = monkey_bs
    _S25._ensure_hubert_model = lambda device: {
        "model": object(), "frame_stride": 320,
        "embedding_dim": 4, "num_layers": 13,
    }
    try:
        out = _S25.post_hoc_verify(
            y_pre, y_post, sr, sid=1, condition="tts",
            repo=_REPO, manifest_path=None,
        )
    finally:
        _S25.extract_frame_embeddings = orig_extract
        _S25.boundary_sharpness = orig_bs
        _S25._ensure_hubert_model = orig_load_model
    # Required fields.
    for k in [
        "stability_metric_pre", "stability_metric_post", "stability_metric_delta",
        "expected_direction", "achieved_direction",
        "lufs_pre", "lufs_post", "delta_lufs",
        "tilt_pre", "tilt_post", "delta_tilt",
        "dyn_pre", "dyn_post", "delta_dyn",
    ]:
        assert k in out, f"missing field {k!r}"


def test_post_hoc_verify_flags_wrong_direction_when_metric_moves_opposite():
    """If stability metric moves opposite the expected_direction, the
    `achieved_direction` should be flagged as 'unexpected'."""
    y_pre = np.zeros(16000, dtype=np.float32)
    y_post = np.zeros(16000, dtype=np.float32)
    # Stub: stability metric pre=0.5, post=0.4 -> delta=-0.1.
    call_state = {"i": 0}
    def fake_bs(frame_times, frame_emb, token_boundaries):
        call_state["i"] += 1
        return (0.0, 0.5 if call_state["i"] % 2 == 1 else 0.4)
    import torch

    fake_h = torch.zeros((1, 10, 4), dtype=torch.float32)
    fake_times = np.arange(10, dtype=np.float32) * 0.02
    _S25.extract_frame_embeddings = (
        lambda model, audio, sample_rate, layers, device="cpu": (fake_h.numpy(), fake_times)
    )
    orig_bs = _S25.boundary_sharpness
    orig_model = _S25._ensure_hubert_model
    _S25.boundary_sharpness = fake_bs
    _S25._ensure_hubert_model = lambda device: {
        "model": object(), "frame_stride": 320, "embedding_dim": 4, "num_layers": 13,
    }
    try:
        out = _S25.post_hoc_verify(
            y_pre, y_post, 16000, sid=1, condition="tts",
            repo=_REPO, manifest_path=None,
            expected_direction="raise_cost",
        )
    finally:
        _S25.boundary_sharpness = orig_bs
        _S25._ensure_hubert_model = orig_model
    # delta = post - pre = 0.4 - 0.5 = -0.1, direction = "lower"
    assert out["stability_metric_delta"] == pytest.approx(-0.1, abs=1e-3)
    assert out["expected_direction"] == "raise_cost"
    assert out["achieved_direction"] == "lower_cost"  # moved opposite
    assert out["achieved_aligns_with_expected"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stability_perturbation.py -v -k "post_hoc"
```

Expected: 2 fails.

- [ ] **Step 3: Implement `post_hoc_verify`**

```python
def post_hoc_verify(
    y_pre: np.ndarray,
    y_post: np.ndarray,
    sr: int,
    sid: int,
    condition: str,
    repo: Path,
    manifest_path: Path | None = None,
    expected_direction: str | None = None,
    device: str = "cuda",
):
    """Recompute stability and acoustic features on pre/post waveforms.

    Returns dict with:
      - stability_metric_pre  / post  / delta
      - expected_direction          (one of raise_cost|lower_cost|None)
      - achieved_direction          (raise_cost|lower_cost|unchanged)
      - achieved_aligns_with_expected (bool|None)
      - lufs_pre/post/delta_lufs, tilt_pre/post/delta_tilt,
        dyn_pre/post/delta_dyn
    """
    if sr != TARGET_SR:
        raise ValueError(f"expected sr={TARGET_SR}; got sr={sr}")

    cache = _ensure_hubert_model(device)
    model = cache["model"]
    frame_stride = cache["frame_stride"]
    layers = [11]

    def _compute_stability(y_audio):
        h, frame_times = extract_frame_embeddings(
            model, y_audio, TARGET_SR, layers, device=device,
        )
        h_l11 = h[0]  # shape (T, D)
        duration_s = float(y_audio.shape[0]) / float(TARGET_SR)
        boundaries = load_token_boundaries(
            sid, condition, repo, manifest_path=manifest_path,
            audio_duration_s=duration_s,
        )
        _bs, ss = boundary_sharpness(frame_times, h_l11, list(boundaries))
        return float(ss)

    ss_pre = _compute_stability(y_pre)
    ss_post = _compute_stability(y_post)
    delta = ss_post - ss_pre

    def _classify_dir(d: float) -> str:
        if abs(d) < 1e-7:
            return "unchanged"
        return "raise_cost" if d > 0 else "lower_cost"

    achieved_direction = _classify_dir(delta)
    if expected_direction is None:
        aligns = None
    elif achieved_direction == "unchanged":
        aligns = None
    else:
        aligns = (achieved_direction == expected_direction)

    lufs_pre = float(compute_lufs(y_pre, sr))
    lufs_post = float(compute_lufs(y_post, sr))
    tilt_pre = float(compute_spectral_tilt(y_pre, sr))
    tilt_post = float(compute_spectral_tilt(y_post, sr))
    dyn_pre = float(compute_energy_env_std(y_pre, sr))
    dyn_post = float(compute_energy_env_std(y_post, sr))

    return {
        "stability_metric_pre": ss_pre,
        "stability_metric_post": ss_post,
        "stability_metric_delta": float(delta),
        "expected_direction": expected_direction,
        "achieved_direction": achieved_direction,
        "achieved_aligns_with_expected": aligns,
        "lufs_pre": lufs_pre,
        "lufs_post": lufs_post,
        "delta_lufs": lufs_post - lufs_pre,
        "tilt_pre": tilt_pre,
        "tilt_post": tilt_post,
        "delta_tilt": tilt_post - tilt_pre,
        "dyn_pre": dyn_pre,
        "dyn_post": dyn_post,
        "delta_dyn": dyn_post - dyn_pre,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stability_perturbation.py -v -k "post_hoc"
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "feat(script25): post_hoc_verify (stability + acoustic drifts)"
```

---

## Task 8: `main()` CLI driver

Wire CLI argument parsing, sample loading, intervention iteration, calling `run_intervention_pipeline` (with post-hoc verification appended to each per-sample record), and aggregate output JSON to `data/wav2sem_analysis/metrics/stability_intervention.json`. Mirror script 23's structure (reuses `load_baseline_results`, `aggregate_deltas`).

**Files:**
- Modify: `scripts/25_stability_perturbation_syncnet.py` (replace `main` stub)
- Test: `tests/test_stability_perturbation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
# ---------------------------------------------------------------------------
# main — dry-run smoke
# ---------------------------------------------------------------------------


def test_main_dry_run_does_not_invoke_ditto_or_syncnet(monkeypatch, tmp_path):
    """A --dry-run invocation must exit cleanly; only the headers and plan
    should be printed. The Ditto/SyncNet/HuBERT modules should remain uncalled.
    """
    monkeypatch.setattr(_S25, "_ensure_hubert_model",
                        lambda device: pytest.fail("PGD should not run on dry-run"))
    # Stub run_intervention_pipeline so it returns a fake result instead of
    # actually calling Ditto. It records the call so we can assert post-hoc
    # verification was performed.
    call_log: list[dict] = []
    def fake_pipeline(intervention, sid, y_tts, y_nat, sr, img_path,
                      audio_out_dir, video_out_dir, eval_out_dir, repo, cfg,
                      run_id, skip_audio, skip_video, skip_syncnet, dry_run):
        call_log.append({
            "intervention": intervention.name, "sid": sid, "dry_run": dry_run,
        })
        # Return a minimal per-sample record on dry-run.
        return {"sync_c": 0.0, "sync_d": 0.0}, "DRY_RUN"
    monkeypatch.setattr(_S25, "run_intervention_pipeline", fake_pipeline)
    # Make a fake run dir so the run-id lookup passes.
    repo = tmp_path
    run_id = "fake_run_xyz"
    run_dir = repo / "runs" / run_id
    (run_dir / "02_tts").mkdir(parents=True)
    (run_dir / "04_eval" / "tts_raw" / "1").mkdir(parents=True)
    (run_dir / "04_eval" / "natural_raw" / "1").mkdir(parents=True)
    (run_dir / "04_eval" / "tts_raw" / "1" / "syncnet.json").write_text(
        json.dumps({"sync_c": 6.0, "sync_d": -1.0, "av_offset": 0})
    )
    (run_dir / "04_eval" / "natural_raw" / "1" / "syncnet.json").write_text(
        json.dumps({"sync_c": 5.0, "sync_d": -1.0, "av_offset": 0})
    )
    # Need samples 1, 2 audio file:
    audio_dir = repo / "data" / "data" / "audio"
    audio_dir.mkdir(parents=True)
    import soundfile as sf
    sr = 16000
    for sid in [1]:
        y = (np.random.default_rng(sid).uniform(-0.3, 0.3, sr * 2)).astype(np.float32)
        sf.write(audio_dir / f"{sid}.wav", y, sr)
        sf.write(run_dir / "02_tts" / f"{sid}.wav", y, sr)
    img_dir = repo / "data" / "data" / "image"
    img_dir.mkdir(parents=True)
    (img_dir / "1.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    monkeypatch.setattr(_S25, "STUDY_SAMPLES", [1])

    import sys as _sys
    _sys.argv = [
        "25_stability_perturbation_syncnet.py",
        "--run-id", run_id,
        "--samples", "1",
        "--dry-run",
        "--output-dir", str(tmp_path / "out"),
        "--device", "cpu",
    ]
    _S25.main()

    assert all(c["dry_run"] for c in call_log), "dry-run must propagate flag"


def test_main_writes_summary_json_when_not_dry_run(monkeypatch, tmp_path):
    """For a non-dry-run run with stubbed pipeline, the summary JSON should
    exist at the expected path."""
    import json

    def fake_pipeline(intervention, sid, y_tts, y_nat, sr, img_path,
                      audio_out_dir, video_out_dir, eval_out_dir, repo, cfg,
                      run_id, skip_audio, skip_video, skip_syncnet, dry_run):
        # Pretend perturbed.
        return {"sync_c": 5.3, "sync_d": -3.0, "av_offset": 0}, "CACHED"
    monkeypatch.setattr(_S25, "run_intervention_pipeline", fake_pipeline)

    monkeypatch.setattr(_S25, "_ensure_hubert_model",
                        lambda device: {"model": None, "frame_stride": 320,
                                         "embedding_dim": 4, "num_layers": 13})

    def fake_extract(model, audio, sr, layers, device="cpu"):
        return np.zeros((1, 50, 4), dtype=np.float32), np.arange(50, dtype=np.float32) * 0.02
    monkeypatch.setattr(_S25, "extract_frame_embeddings", fake_extract)
    monkeypatch.setattr(_S25, "boundary_sharpness", lambda ft, h, b: (0.0, 0.15))

    repo = tmp_path
    run_id = "fake_run_xyz"
    run_dir = repo / "runs" / run_id
    (run_dir / "02_tts").mkdir(parents=True)
    (run_dir / "04_eval" / "tts_raw" / "1").mkdir(parents=True)
    (run_dir / "04_eval" / "natural_raw" / "1").mkdir(parents=True)
    (run_dir / "04_eval" / "tts_raw" / "1" / "syncnet.json").write_text(
        json.dumps({"sync_c": 6.0, "sync_d": -1.0, "av_offset": 0})
    )
    (run_dir / "04_eval" / "natural_raw" / "1" / "syncnet.json").write_text(
        json.dumps({"sync_c": 5.0, "sync_d": -1.0, "av_offset": 0})
    )
    audio_dir = repo / "data" / "data" / "audio"
    audio_dir.mkdir(parents=True)
    import soundfile as sf
    for sid in [1]:
        y = (np.random.default_rng(sid).uniform(-0.3, 0.3, 16000 * 2)).astype(np.float32)
        sf.write(audio_dir / f"{sid}.wav", y, 16000)
        sf.write(run_dir / "02_tts" / f"{sid}.wav", y, 16000)
    img_dir = repo / "data" / "data" / "image"
    img_dir.mkdir(parents=True)
    (img_dir / "1.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    monkeypatch.setattr(_S25, "STUDY_SAMPLES", [1])

    out_dir = tmp_path / "out"
    import sys as _sys
    _sys.argv = [
        "25_stability_perturbation_syncnet.py",
        "--run-id", run_id,
        "--samples", "1",
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--pgd-steps", "3",
    ]
    # Need a manifest stub — point --manifest at a nonexistent file so
    # load_token_boundaries returns empty (and post_hoc falls back).
    _sys.argv += ["--manifest", str(tmp_path / "nonexistent.json")]

    _S25.main()

    out_path = out_dir / "stability_intervention.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "interventions" in data
    assert len(data["interventions"]) == 3
    for name in ("stability_adj_tts", "stability_adj_nat", "random_noise_tts"):
        assert name in data["interventions"]


import json  # placed at end of test file to keep step 1 self-contained
```

Note: the `import json` line at the end of the test file is intentional; placing it at the end is fine since Python’s module-loading model makes it visible to all preceding tests too. Alternatively, move it to the top imports block. We'll move it during the real edit (i.e. tests will have `import json` at the top in their final form).

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_stability_perturbation.py -v -k "main"
```

Expected: 2 fails (AttributeError on `_sys` or function not implemented).

- [ ] **Step 3: Implement `main`**

Replace the stub with:

```python
def main() -> None:
    """CLI entry point: dispatch the 3 stability interventions over samples.

    Mirrors script 23's structure: load baselines, iterate cells × samples,
    each cell calls run_intervention_pipeline with the cell's transform;
    post-hoc verification is appended to each per-sample record so the
    output JSON gives script 26 (the verdict classifier, arriving later)
    enough material to decide STABILITY_CAUSAL / REJECTED / INCONCLUSIVE.
    """
    ap = argparse.ArgumentParser(
        description="Stability-targeted adversarial perturbation (script 25)"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--samples",
        default=",".join(str(s) for s in STUDY_SAMPLES),
        help="Comma-separated sample IDs (default: STUDY_SAMPLES)",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None,
                   help="Token-boundary manifest JSON (default: data/wav2sem_analysis/manifest/alignment.json)")
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--pgd-steps", type=int, default=50)
    ap.add_argument("--pgd-restarts", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda",
                   help="cpu or cuda (default: cuda)")
    ap.add_argument("--no-cache-audio", action="store_true")
    ap.add_argument("--no-cache-video", action="store_true")
    ap.add_argument("--no-cache-syncnet", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.repo = _REPO
    if args.output_dir is None:
        args.output_dir = ensure_output_dirs()["metrics"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.repo / "runs" / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        return

    cfg_path = args.repo / "scripts" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    sample_ids = [int(s.strip()) for s in args.samples.split(",") if s.strip()]

    interventions = _build_interventions(args)
    eval_base = run_dir / "04_eval"
    baseline_conds = sorted({iv.baseline_cond for iv in interventions})
    baselines = load_baseline_results(eval_base, baseline_conds)

    nat_audio_dir = args.repo / "data" / "data" / "audio"
    tts_audio_dir = run_dir / "02_tts"
    img_dir = args.repo / "data" / "data" / "image"

    if not tts_audio_dir.is_dir():
        print(f"ERROR: TTS audio dir not found: {tts_audio_dir}")
        return

    # Preload audio (so dry-run doesn't need HuBERT).
    natural_audio: dict[int, np.ndarray] = {}
    tts_audio: dict[int, np.ndarray] = {}
    loaded_ids: list[int] = []
    for sid in sample_ids:
        nat_p = nat_audio_dir / f"{sid}.wav"
        tts_p = tts_audio_dir / f"{sid}.wav"
        if nat_p.exists() and tts_p.exists():
            natural_audio[sid], _ = load_audio_mono(nat_p, target_sr=TARGET_SR)
            tts_audio[sid], _ = load_audio_mono(tts_p, target_sr=TARGET_SR)
            loaded_ids.append(sid)
        else:
            print(f"WARN: sample {sid} missing audio (nat={nat_p.exists()}, tts={tts_p.exists()})")

    sr = TARGET_SR

    ditto_intv_base = run_dir / "03_ditto" / "interventions"
    eval_intv_base = run_dir / "04_eval" / "interventions"

    print(f"[script25] run_id={args.run_id}")
    print(f"[script25] samples loaded: {loaded_ids}")
    print(f"[script25] cells: {[iv.name for iv in interventions]}")
    print(f"[script25] pgd: eps={args.eps}, alpha={args.alpha}, K={args.pgd_steps}, device={args.device}")
    print(f"[script25] baselines: {[(c, len(v)) for c, v in baselines.items()]}")
    if args.dry_run:
        print("[script25] DRY-RUN — no Ditto/SyncNet invocation\n")

    results: dict[str, dict[int, dict]] = {iv.name: {} for iv in interventions}
    post_hoc_records: dict[str, dict[int, dict]] = {iv.name: {} for iv in interventions}
    failures: list[dict] = []
    t0 = time.monotonic()

    for iv in interventions:
        print(f"\n=== {iv.name} ({iv.transform_description}) ===")
        for sid in loaded_ids:
            img_path = img_dir / f"{sid}.png"
            if not img_path.exists():
                failures.append({"intervention": iv.name, "sample_id": sid, "error": "image missing"})
                continue

            audio_out = ditto_intv_base / iv.name
            video_out = ditto_intv_base / iv.name
            eval_out = eval_intv_base / iv.name

            # Save unperturbed source for post-hoc verification. The intervention
            # transform itself may overwrite the cached audio, so compute
            # verification BEFORE the pipeline invocation using the *pre* source.
            if iv.source == "tts":
                y_pre = tts_audio[sid]
                condition = "tts"
            else:
                y_pre = natural_audio[sid]
                condition = "natural"
            expected_direction = {
                "stability_adj_tts": "raise_cost",
                "stability_adj_nat": "lower_cost",
                "random_noise_tts": None,
            }[iv.name]

            res, status = run_intervention_pipeline(
                intervention=iv,
                sid=sid,
                y_tts=tts_audio[sid],
                y_nat=natural_audio[sid],
                sr=sr,
                img_path=img_path,
                audio_out_dir=audio_out,
                video_out_dir=video_out,
                eval_out_dir=eval_out,
                repo=args.repo,
                cfg=cfg,
                run_id=args.run_id,
                skip_audio=not args.no_cache_audio,
                skip_video=not args.no_cache_video,
                skip_syncnet=not args.no_cache_syncnet,
                dry_run=args.dry_run,
            )

            if res is None:
                failures.append({"intervention": iv.name, "sample_id": sid, "error": status})
                print(f"  sample {sid}: FAIL — {status}")
                continue

            # Load perturbed audio for post-hoc verification. The transformed
            # audio was written to audio_out / f"{sid}.wav" by the pipeline.
            if not args.dry_run:
                perturbed_path = audio_out / f"{sid}.wav"
                if perturbed_path.exists():
                    y_post, _ = load_audio_mono(perturbed_path, target_sr=TARGET_SR)
                else:
                    y_post = y_pre
                post_hoc = post_hoc_verify(
                    y_pre=y_pre, y_post=y_post, sr=sr, sid=sid, condition=condition,
                    repo=args.repo, manifest_path=args.manifest,
                    expected_direction=expected_direction, device=args.device,
                )
                res["post_hoc"] = post_hoc
                post_hoc_records[iv.name][sid] = post_hoc

            results[iv.name][sid] = res

            c = res.get("sync_c", 0.0)
            base = baselines.get(iv.baseline_cond, {}).get(sid, {})
            bc = base.get("sync_c")
            if bc is not None:
                print(
                    f"  sample {sid}: Sync-C={c:.3f} (base={bc:.3f}, Δ={c - bc:+.3f})"
                    f" — {status}"
                )
            else:
                print(f"  sample {sid}: Sync-C={c:.3f} (no baseline) — {status}")

    elapsed = time.monotonic() - t0
    print(
        f"\n[script25] completed in {elapsed:.0f}s; "
        f"{sum(len(v) for v in results.values())} ok, {len(failures)} failures"
    )

    if args.dry_run:
        print("[script25] DRY-RUN — no JSON written")
        return

    summary = aggregate_deltas(results, baselines, interventions)

    output = {
        "run_id": args.run_id,
        "samples": loaded_ids,
        "baselines": {
            cond: {str(sid): vals for sid, vals in baselines.get(cond, {}).items()}
            for cond in baseline_conds
        },
        "interventions": summary,
        "post_hoc": post_hoc_records,
        "failed": failures,
        "elapsed_s": round(elapsed, 1),
        "pgd_params": {
            "eps": args.eps,
            "alpha": args.alpha,
            "K": args.pgd_steps,
            "restarts": args.pgd_restarts,
        },
        "verdict": "TBD",  # filled by script 26 downstream
    }

    out_path = args.output_dir / "stability_intervention.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"[script25] results -> {out_path}")

    print("\n=== Stability Intervention Summary ===")
    for iv_name, s in summary.items():
        if s.get("n_samples"):
            print(
                f"{iv_name:<22} n={s['n_samples']:>2}  "
                f"ΔSync-C={s['mean_delta_c']:+.4f}  ΔSync-D={s['mean_delta_d']:+.4f}"
            )
```

Note: the full `post_hoc_records` per-sample dict gets stored alongside the aggregated `interventions` summary in the output JSON — script 26 will read both. Post-hoc per-sample records live under the top-level `post_hoc` key (not under `interventions`) since `aggregate_deltas` from script 22 produces per-intervention *summary* stats only.

- [ ] **Step 4: Re-run main tests to verify they pass**

Ensure `import json` is at the top of `tests/test_stability_perturbation.py` (add it during this step if missing — the test fixture writes JSON for syncnet baseline files):

```bash
pytest tests/test_stability_perturbation.py -v -k "main"
```

Expected: 2 PASS.

- [ ] **Step 5: Run ALL script-25 tests to verify nothing regressed**

```bash
pytest tests/test_stability_perturbation.py -v
```

Expected: all tests PASS (import + stability_loss + boundary + random_noise + pgd + build_interventions + post_hoc + main).

- [ ] **Step 6: Commit**

```bash
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "feat(script25): main() CLI driver with --dry-run support and post_hoc re-verify"
```

---

## Task 9: Full-suite test pass + local smoke

**Files:**
- (No file changes; just verification.)

- [ ] **Step 1: Run the full local pytest suite**

```bash
pytest tests/ -v
```

Expected: all ≥219 prior tests still PASS plus new script-25 tests PASS. No regression.

- [ ] **Step 2: Inspect output for any test-collection warnings**

```bash
pytest tests/ -v 2>&1 | tail -30
```

Expected: no collection errors, no skipped tests caused by our changes.

- [ ] **Step 3: Commit (if test file changes during step-9 cleanup)**

Only if Step 1 forced adding `import json` at the top of the test file (which might have been deferred during Task 8). Otherwise skip this commit.

```bash
git add tests/test_stability_perturbation.py
git commit -m "test(script25): finalize imports and test-suite pass"
```

---

## Task 10: Sync to server and run smoke (1 sample)

**Files:**
- (No file changes; just server execution.)

- [ ] **Step 1: rsync script + accompanying test to server**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e scp -P 26325 -o StrictHostKeyChecking=no \
  scripts/25_stability_perturbation_syncnet.py \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/repos/tts-exp/scripts/

SSHPASS='U6UYPaI+/B3X' sshpass -e scp -P 26325 -o StrictHostKeyChecking=no \
  tests/test_stability_perturbation.py \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/repos/tts-exp/tests/
```

Manifest JSON is already on the server from script 16's run — verify in Step 2 before deciding whether to upload.

- [ ] **Step 2: Ensure alignment.json exists on server**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e ssh -p 26325 -o StrictHostKeyChecking=no root@connect.westd.seetacloud.com \
  "ls -la /root/autodl-tmp/repos/tts-exp/data/wav2sem_analysis/manifest/"
```

If `alignment.json` is missing, upload the local one. Skip upload if it's already present (it should be — script 16 ran on server to produce it).

- [ ] **Step 3: Run pytest on the server to verify imports + 5 stubs work in the GPU env**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e ssh -p 26325 -o StrictHostKeyChecking=no root@connect.westd.seetacloud.com \
  "cd /root/autodl-tmp/repos/tts-exp && source /root/miniconda3/envs/syncnet/bin/activate \
   && pytest tests/test_stability_perturbation.py -v 2>&1 | tail -50"
```

Expected: all tests PASS on server.

- [ ] **Step 4: Dry-run smoke on server (1 cell iteration, no pipeline invocation)**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e ssh -p 26325 -o StrictHostKeyChecking=no root@connect.westd.seetacloud.com \
  "cd /root/autodl-tmp/repos/tts-exp && source /root/miniconda3/envs/ditto/bin/activate \
   && python scripts/25_stability_perturbation_syncnet.py \
        --run-id r2_faster_qwen3_20260707T145233Z \
        --samples 1 \
        --dry-run"
```

Expected: prints `[script25] DRY-RUN — no Ditto/SyncNet invocation` and exits 0. No HuBERT model load since the transform is called only inside `run_intervention_pipeline` (which respects dry-run). No `stability_intervention.json` written.

- [ ] **Step 5: Live smoke on 1 sample (HuBERT forward+backward on GPU + Ditto + SyncNet)**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e ssh -t -p 26325 -o StrictHostKeyChecking=no root@connect.westd.seetacloud.com \
  "cd /root/autodl-tmp/repos/tts-exp && source /root/miniconda3/envs/ditto/bin/activate \
   && python scripts/25_stability_perturbation_syncnet.py \
        --run-id r2_faster_qwen3_20260707T145233Z \
        --samples 1 \
        --pgd-steps 10 \
        --device cuda"
```

Expected: complete in <15 min, writes `data/wav2sem_analysis/metrics/stability_intervention.json` with 3 cell entries (each containing per-sample dict for sample 1). Each entry's `post_hoc` block contains `stability_metric_pre`, `stability_metric_post`, etc.

Issues to watch for:
- Out-of-memory on GPU (Ditto inference should free its allocations; HuBERT forward+backward over `eps`-perturbed 16 kHz audio is ~32MB activations, well within 4080's 16GB).
- HuBERT model failing to load due to offline-mode cache miss — script 15 already cached weights on server from earlier runs.
- `transformers` version mismatch on `output_hidden_states=True` — already exercised by script 15 so this should not surface.

- [ ] **Step 6: Sanity-check post_hoc direction flags in the smoke output**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e ssh -p 26325 -o StrictHostKeyChecking=no root@connect.westd.seetacloud.com \
  "cat /root/autodl-tmp/repos/tts-exp/data/wav2sem_analysis/metrics/stability_intervention.json \
   | python -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d[\"post_hoc\"], indent=2))'"
```

Expected:
- `stability_adj_tts[1].stability_metric_delta > 0` (cost went up) and `achieved_aligns_with_expected == True`.
- `stability_adj_nat[1].stability_metric_delta < 0` (cost went down) and `achieved_aligns_with_expected == True`.
- `random_noise_tts[1]`: no `expected_direction` (None) and `achieved_direction` may be either sign.
- All acoustic drifts (`delta_lufs`, `delta_tilt`, `delta_dyn`) within ±0.5 dB / ±0.3 dB/oct / ±1.0 dB respectively.

If direction is wrong on either cell, the sign convention has a bug — recheck Task 5.

- [ ] **Step 7: Commit any local arg-tweaks/compatibility fixes (likely none)**

```bash
git status
# If any local adjustments needed to make server work:
git add scripts/25_stability_perturbation_syncnet.py tests/test_stability_perturbation.py
git commit -m "fix(script25): server-compatibility adjustments from 1-sample smoke"
```

---

## Task 11: Full server run (9 samples × 3 cells)

**Files:**
- (No file changes; full server execution.)

- [ ] **Step 1: Launch full 9-sample run in a tmux session on the server (long-running)**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e ssh -t -p 26325 -o StrictHostKeyChecking=no root@connect.westd.seetacloud.com \
  "tmux new-session -d -s s25 \
   'cd /root/autodl-tmp/repos/tts-exp && source /root/miniconda3/envs/ditto/bin/activate \
     && python scripts/25_stability_perturbation_syncnet.py \
            --run-id r2_faster_qwen3_20260707T145233Z \
            --pgd-steps 50 \
            --device cuda 2>&1 | tee /tmp/s25.log'"
```

Sample IDs default to STUDY_SAMPLES (9 samples, excl. sample 9).

- [ ] **Step 2: Poll the log every ~10 minutes until completion**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e ssh -p 26325 -o StrictHostKeyChecking=no root@connect.westd.seetacloud.com \
  "tail -30 /tmp/s25.log; echo === tmux-status ===; tmux ls"
```

Expected completion in ~2h (27 cells × ~5min each). Look for `[script25] completed in NNs; 27 ok, 0 failures`.

- [ ] **Step 3: Once complete, verify the JSON file size and basic structure**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e ssh -p 26325 -o StrictHostKeyChecking=no root@connect.westd.seetacloud.com \
  "ls -lh /root/autodl-tmp/repos/tts-exp/data/wav2sem_analysis/metrics/stability_intervention.json \
   && python -c 'import json; d=json.load(open(\"/root/autodl-tmp/repos/tts-exp/data/wav2sem_analysis/metrics/stability_intervention.json\")); \
                  print(len(d[\"interventions\"]), \"cells\"); \
                  [print(k, v.get(\"n_samples\"), \"mean ΔC\", v.get(\"mean_delta_c\")) for k,v in d[\"interventions\"].items()]'"
```

Expected: 3 cells, each with n_samples=9, mean ΔSync-C values printed. Zero failures.

- [ ] **Step 4: Pull the result artifact back to local for downstream analysis**

```bash
SSHPASS='U6UYPaI+/B3X' sshpass -e scp -P 26325 -o StrictHostKeyChecking=no \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/repos/tts-exp/data/wav2sem_analysis/metrics/stability_intervention.json \
  data/wav2sem_analysis/metrics/stability_intervention.json
```

- [ ] **Step 5: Commit the result artifact + log lines**

```bash
git add data/wav2sem_analysis/metrics/stability_intervention.json
git commit -m "data(script25): full 9-sample × 3-cell stability intervention results"
```

---

## Task 12: Update docs (execution log + mechanism report)

Mirror the documentation structure added for script 23/24: append a Stability Intervention section to the Phase 2 execution log, and update the mechanism report's findings and candidate table.

**Files:**
- Modify: `docs/experiments/phase2_execution_20260718.md`
- Modify: `docs/experiments/tts_tfg_mechanism_report.md`

- [ ] **Step 1: Read the current execution-log structure to mirror conventions**

```bash
# Skip if you already recall the file. Read docs/experiments/phase2_execution_20260718.md
```

- [ ] **Step 2: Append a "Stability Intervention" section to the execution log**

Append ~80 lines mirroring the structure of the Identity Control + Identity-Corrected sections: (1) hypotheses, (2) cells + sample counts, (3) PGD params, (4) per-cell aggregated Sync-C deltas, (5) post-hoc verification flag counts (how many 7/9 correctly-aligned), (6) acoustic-drift flag counts (how many within ±0.5 dB LUFS / etc.), (7) plain-language conclusion + "see script 26 for the formal verdict."

Wait for the actual data from Task 11's run before finalising numbers. Use placeholder values ONLY in the form `<TBD:numbers>` until the pull in Step 11.4 finishes, then replace them.

- [ ] **Step 3: Update the candidate table in the mechanism report**

Find the row describing "HuBERT `segment_stability` (layer 11)" in `tts_tfg_mechanism_report.md` and update its "Evidence Strength" column with either `Strong (Phase 2 stability intervention — script 25 confirms/rejects/inconclusive)` depending on the verdict. Add a new row entry for script 26's verdict code once script 26 lands.

- [ ] **Step 4: Commit doc updates**

```bash
git add docs/experiments/phase2_execution_20260718.md docs/experiments/tts_tfg_mechanism_report.md
git commit -m "docs(script25): log stability intervention + update mechanism report"
```

---

## Out-of-Scope (tracked for plan-26)

The statistical test that produces the `STABILITY_CAUSAL` / `STABILITY_REJECTED` / `INCONCLUSIVE` verdict — `tests 1, 2, 3` of the spec — lives in script 26, which is its own plan after the script 25 run results are in hand. Reasons: 1) the test expectations depend on observed post-hoc directionality; 2) it's a 200-line paired-t analysis mirroring script 24 with its own unit tests. Writing it now risks a dimension mismatch between the 3-tier ladder's expected means and what the 27-cell run actually produced.

Acceptance gate for script 25 to be considered done:
- All 12–16 unit tests pass.
- 9-sample × 3-cell server run completes with zero failures.
- `data/wav2sem_analysis/metrics/stability_intervention.json` exists, has all 3 interventions, all 9 samples, and post_hoc verification for each cell.
- At least 7/9 samples per PGD cell show `achieved_aligns_with_expected == True`.
- Acoustic drifts within documented thresholds for ≥7/9 samples per cell.
- Docs updated; commit history clean.

Once accepted, script 26 will be designed in a separate brainstorming + plan cycle.

---

## Self-review notes (for the implementer)

1. **Direction semantics:** re-read the spec's "Sign-convention summary" section every time you touch `direction`. `raise_cost` ⟺ cost metric goes up ⟺ less stable. `lower_cost` ⟺ cost metric goes down ⟺ more stable. TTS cell = raise_cost; nat cell = lower_cost.
2. **`pgd_perturb` sign-factor recurrence:** `direction_factor = -1 if direction == "raise_cost" else +1`. PGD step = `delta -= direction_factor * alpha * sign(grad)`. This descends `direction_factor * L`. For `raise_cost` (`-1`), step descends `-L` = ascends `L` → cost up. ✓
3. **Boundary indices are frame indices, not times.** `searchsorted(frame_times, dt_boundary) - 1` is the "before-boundary" frame. The `boundary_sharpness` reference does the same internally.
4. **`extract_frame_embeddings` returns `(embeddings, frame_times)` with shape (n_layers, T, D).** Post-hoc takes `h[0]` (the only requested layer-11). Don't pass `layers=[11]` thinking `h` will be shape (T, D); it has the leading layer axis.
5. **manifest path:** script 25 accepts `--manifest` so that tests can point at synthetic JSON; production runs use the default `data/wav2sem_analysis/manifest/alignment.json`. If a sample is missing from the manifest, fallback to uniform 50ms segments (this matters for `random_noise_tts` too — its `post_hoc_verify` uses the same loader for consistency).
6. **Cache reuse:** Ditto + SyncNet cache directories already exist from scripts 22/23. When caching is on, `skip_audio=True` etc. — the cell's transformed audio is reused from disk if present. Don't rewrite the cache on dry-run.
7. **`run_intervention_pipeline` calls `transform(y_tts, y_nat, sr, sid)`.** For natural-source interventions, *it still passes the TTS audio as the first arg*. The transform itself picks `y_nat` from `condition="natural"`. The pipeline doesn't know which waveform we perturb; only the transform closure does.
8. **Failure modes TFG-side:** face-detection failure on sample 9 was confirmed during script 23 work. STUDY_SAMPLES already excludes 9. Same exclusion applies here.