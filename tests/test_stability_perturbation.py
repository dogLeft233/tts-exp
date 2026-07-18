"""Unit tests for scripts/25_stability_perturbation_syncnet.py.

These tests exercise the stability-targeted PGD transform, the random-noise
control transform, token-boundary loading, the post-hoc verification, and
the intervention registry. GPU/HuBERT-forward paths are mocked where needed.
"""

from __future__ import annotations

import argparse
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
    p.write_text(json.dumps(manifest))
    repo = tmp_path

    b_tts = _S25.load_token_boundaries(1, "tts", repo, manifest_path=p)
    b_nat = _S25.load_token_boundaries(1, "natural", repo, manifest_path=p)
    assert list(b_tts) == pytest.approx([0.20, 0.45], abs=1e-6)
    assert list(b_nat) == pytest.approx([0.30], abs=1e-6)


def test_load_token_boundaries_falls_back_when_sample_missing(tmp_path):

    manifest = {"samples": [
        {"sample_id": 1, "condition": "tts", "tokens": [
            {"start_s": 0.0, "end_s": 0.30},
        ]}
    ]}
    p = tmp_path / "alignment.json"
    p.write_text(json.dumps(manifest))

    boundaries = _S25.load_token_boundaries(
        2, "tts", tmp_path, manifest_path=p, audio_duration_s=1.0,
    )
    assert len(boundaries) > 0
    assert boundaries[0] == pytest.approx(0.05, abs=1e-6)


def test_load_token_boundaries_returns_empty_when_no_duration_for_fallback(tmp_path):
    """If a sample is missing from the manifest AND no audio_duration_s is
    provided, return an empty array so downstream code skips stability loss
    over zero within-segment pairs."""
    manifest = {"samples": []}
    p = tmp_path / "alignment.json"
    p.write_text(json.dumps(manifest))
    out = _S25.load_token_boundaries(
        2, "tts", tmp_path, manifest_path=p, audio_duration_s=None,
    )
    assert isinstance(out, np.ndarray)
    assert out.size == 0


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
    y = np.ones(8000, dtype=np.float32)
    y_pulse = _S25.random_sign_noise_transform(y, None, 16000, 1, eps=0.005)
    assert y_pulse.min() >= -1.0 - 1e-8
    assert y_pulse.max() <= 1.0 + 1e-8


def test_random_sign_noise_different_seed_each_call_gives_different_delta():
    y = np.zeros(4096, dtype=np.float32)
    a = _S25.random_sign_noise_transform(y, None, 16000, 1, eps=0.005)
    b = _S25.random_sign_noise_transform(y, None, 16000, 2, eps=0.005)
    assert not np.allclose(a, b)


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
        self.weight = torch.randn(D, 1, generator=g) * 0.1
        self.bias = torch.randn(D, generator=g) * 0.05
        self.config = self._StubConfig()

    class _StubConfig:
        conv_stride = [5, 2, 2, 2, 2, 2, 2]
        num_hidden_layers = 13

    def __call__(self, waveform, output_hidden_states=True):
        import torch

        n = waveform.shape[-1]
        step = max(1, n // self.T_out)
        frames = waveform[..., :step * self.T_out].reshape(1, self.T_out, step)
        frames_mean = frames.mean(dim=2, keepdim=False).unsqueeze(-1)
        h_l12 = frames_mean @ self.weight.t() + self.bias
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
    def compute_loss(audio_np):
        with torch.no_grad():
            x = torch.from_numpy(audio_np).float().unsqueeze(0)
            out = model(x)
            h_l11 = out.hidden_states[12]
            n_frames = h_l11.shape[1]
            ft = np.arange(n_frames, dtype=np.float32) * (
                frame_stride / _S25.TARGET_SR
            )
            bi = np.searchsorted(ft, dt_boundaries) - 1
            bi = bi[(bi >= 0) & (bi < n_frames - 1)]
            bi = bi.astype(np.int64)
            return float(_S25.stability_loss(h_l11, ft, bi))

    loss_pre = compute_loss(y)

    out_raise = _S25.pgd_perturb(
        y, model, frame_stride, dt_boundaries,
        eps=0.05, alpha=0.001, K=31, device="cpu", direction="raise_cost",
    )
    loss_post_raise = compute_loss(out_raise)
    assert loss_post_raise > loss_pre, "raise_cost should increase stability_loss"

    out_lower = _S25.pgd_perturb(
        y, model, frame_stride, dt_boundaries,
        eps=0.05, alpha=0.001, K=31, device="cpu", direction="lower_cost",
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


# ---------------------------------------------------------------------------
# _build_interventions
# ---------------------------------------------------------------------------


def test_build_interventions_returns_three_cells(monkeypatch):
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
    assert cells[0].source == "tts"
    assert cells[0].baseline_cond == "tts_raw"
    assert cells[1].source == "natural"
    assert cells[1].baseline_cond == "natural_raw"
    assert cells[2].source == "tts"
    assert cells[2].baseline_cond == "tts_raw"


def test_build_interventions_random_noise_uses_tts_source(monkeypatch):
    monkeypatch.setattr(
        _S25, "pgd_stability_transform",
        lambda **kw: (lambda y_t, y_n, sr, sid: y_t * 2),
    )
    ns = argparse.Namespace(
        eps=0.005, alpha=0.001, pgd_steps=50, pgd_restarts=1,
        device="cpu", repo=_REPO,
        manifest=None,
    )
    cells = _S25._build_interventions(ns)
    sentinel_stubs = {"called": 0}

    def fake(y_src, _y_other, _sr, _sid, eps=0.005):
        sentinel_stubs["called"] += 1
        return y_src + 0.001

    monkeypatch.setattr(_S25, "random_sign_noise_transform", fake)
    y_t = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    y_n = np.array([-1.0, 1.0, -0.5], dtype=np.float32)
    out = cells[2].transform(y_t, y_n, 16000, 5)
    assert sentinel_stubs["called"] == 1
    assert out.shape == y_t.shape
    np.testing.assert_allclose(out, y_t + 0.001, atol=1e-6)


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
    monkeypatch.setattr(_S25, "_REPO", tmp_path)
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
    # Need samples 1 audio file:
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
    monkeypatch.setattr(_S25, "_REPO", tmp_path)

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
