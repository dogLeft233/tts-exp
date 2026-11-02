"""Unit tests for scripts/41_video_quality_metrics.py and 43_aggregate_video_metrics.py."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


metrics = _load("vqm", REPO / "scripts" / "41_video_quality_metrics.py")
agg = _load("vqm_agg", REPO / "scripts" / "43_aggregate_video_metrics.py")


def test_sample_indices_uniform_and_bounded():
    idx = metrics.sample_indices(100, 10)
    assert len(idx) == 10
    assert idx[0] == 0 and idx[-1] == 99
    assert np.all(np.diff(idx) > 0)
    # fewer frames than requested -> all frames, no crash
    assert len(metrics.sample_indices(5, 10)) == 5
    assert len(metrics.sample_indices(0, 10)) == 0


def test_ssim_psnr_identity_is_perfect():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    ssim, psnr = metrics.ssim_psnr(img, img)
    assert ssim == pytest.approx(1.0)
    assert psnr == float("inf")


def test_ssim_psnr_degrades_with_noise():
    rng = np.random.default_rng(1)
    base = np.full((64, 64, 3), 128, dtype=np.uint8)
    noisy = np.clip(base.astype(np.int16) + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)
    ssim, psnr = metrics.ssim_psnr(base, noisy)
    assert ssim < 0.99
    assert 10 < psnr < 40


def test_frechet_distance_zero_for_same_distribution():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 8))
    assert metrics.frechet_distance(x, x) < 1e-6


def test_frechet_distance_grows_with_shift():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 8))
    y = rng.normal(size=(400, 8)) + 2.0
    assert metrics.frechet_distance(x, y) > metrics.frechet_distance(x, rng.normal(size=(400, 8)))


def test_polynomial_mmd2_zero_for_identical_samples():
    # Small-magnitude features keep the polynomial kernel near 1, so the
    # unbiased (diagonal-excluded) estimator must collapse to ~0 for x == y.
    rng = np.random.default_rng(4)
    x = 0.1 * rng.normal(size=(200, 6))
    assert abs(metrics.polynomial_mmd2(x, x)) < 0.01
    # ... and must grow once the two samples are separated
    assert metrics.polynomial_mmd2(x, x + 0.5) > 0.01


def test_paired_stats_delta_direction():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = a + 1.0
    st = agg.paired_stats(a, b)
    assert st["delta"] == pytest.approx(1.0)
    assert st["win_rate_b_gt_a"] == 1.0
    # fewer than 5 pairs -> no bootstrap CI
    st2 = agg.paired_stats(a[:3], b[:3])
    assert "delta_ci" not in st2


def test_derive_stride_normalisation():
    rows = [{"n_frames_total": 200, "n_frames_sampled": 100,
             "frame_diff_l1": 2.0, "lpips_temporal": 0.5}]
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "per_video_shard0of1.jsonl"
        p.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        out = agg.load_rows(Path(td))
    assert out[0]["sampled_stride"] == pytest.approx(2.0)
    assert out[0]["diff_per_frame"] == pytest.approx(1.0)
    assert out[0]["lpips_per_frame"] == pytest.approx(0.25)


def test_relative_videos_resolve_from_selected_repository(tmp_path, monkeypatch):
    import json
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": [
        {"group": "natural", "item": "1", "video": "results/1.mp4"}
    ]}))
    seen = []
    monkeypatch.setattr(metrics, "MetricModels", lambda *a, **kw: object())
    def score(path, *args):
        seen.append(path)
        return {"n_frames_total": 2, "n_frames_sampled": 2}
    monkeypatch.setattr(metrics, "score_video", score)
    monkeypatch.chdir(tmp_path)
    assert metrics.main(["--manifest", str(manifest), "--out", str(tmp_path / "out"),
                         "--repo-root", str(repo), "--no-iqa", "--no-distribution"]) == 0
    assert seen == [repo / "results/1.mp4"]


def test_video_decoder_samples_actual_clip(tmp_path):
    import cv2
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 64))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((64, 64, 3), i * 15, dtype=np.uint8))
    writer.release()
    frames, fps, total = metrics.read_video_frames(path, 4)
    assert len(frames) == 4 and total == 12
    assert fps == pytest.approx(25)
    assert frames[0].mean() < frames[-1].mean()
