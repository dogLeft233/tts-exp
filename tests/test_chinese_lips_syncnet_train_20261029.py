"""Intervention invariants and timing support, independent of any model outcome."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/experiments/chinese_lips_syncnet_train_20261029.py"
SPEC = importlib.util.spec_from_file_location("lips_train", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_exact_reweight_preserves_every_stratum_mass_and_identical_support():
    # 25/75 and 50/50 both have sufficient common support; a pure stratum does not.
    strata = [[6, 1, 1, 5, 5]] * 8 + [[-6, 1, 1, 5, 5]] * 4 + [[7, 1, 1, 5, 5]] * 4
    labels = [0, 0, 1, 1, 1, 1, 1, 1] + [0, 0, 1, 1] + [0, 0, 0, 0]
    keep, weights, audit = M.balancing_weights(strata, labels)
    assert keep.tolist() == [True] * 12 + [False] * 4
    for key in {tuple(s) for s in np.asarray(strata)[keep]}:
        mask = np.all(np.asarray(strata) == key, axis=1)
        assert weights[mask].sum() == pytest.approx(mask.sum())
        assert weights[mask & (np.asarray(labels) == 1)].sum() == pytest.approx(mask.sum() / 2)
    assert weights[keep].mean() == pytest.approx(1)
    assert weights.max() <= 4
    assert sum(x["retained"] for x in audit) == 2


def test_rare_class_excluded_in_both_arms_rather_than_clipped():
    keep, weights, audit = M.balancing_weights([[6]] * 9, [0] * 8 + [1], max_weight=4)
    assert not keep.any() and not weights.any()
    assert audit[0]["retained"] is False


def test_whole_window_phone_control_detects_edges_not_only_center(tmp_path):
    # Central phone is identical but only 40% of the full temporal sequence matches.
    a = ["b"] * 3 + ["a"] * 4 + ["p"] * 3
    b = ["t"] * 3 + ["a"] * 4 + ["s"] * 3
    assert M.phone_similarity(a, b) == .4
    assert M.phones_at_window([(0., .1, "b"), (.1, .2, "a")], 0) == ["b"] * 5 + ["a"] * 5
    assert M.phones_at_window([(0., .1, "b")], 0) is None
    path = tmp_path / "phone.json"
    path.write_text(json.dumps([[0, .1, "b"], [.1, .2, "a"]]), encoding="utf-8")
    assert M.phone_spans({"id": "x", "phone_spans_path": str(path)}) == [(0, .1, "b"), (.1, .2, "a")]


def test_manifest_rejects_speaker_leakage(tmp_path):
    basic = dict(speaker="same", video_path="a", audio_path="a", mfcc_path="a", frame_count=80)
    rows = [dict(basic, id="a", split="train"), dict(basic, id="b", split="val")]
    path = tmp_path / "clips.json"
    path.write_text(json.dumps({"metadata": {}, "rows": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="speaker leakage"):
        M.read_manifest(path)


def test_complete_real_lag_support_and_pcm_energy_units():
    assert M.max_window_start({"frame_count": 80}, 320, 51200) == 74
    assert M.max_window_start({"frame_count": 80}, 320, 51440) == 75
    assert M.max_window_start({"frame_count": 80}, 319, 51200) == 74
    assert M.max_window_start({"frame_count": 25}, 100, 16100) == 19
    assert min(M.LAGS) == -25 and max(M.LAGS) == 25 and len(M.LAGS) == 40
    assert all(abs(lag) >= 6 for lag in M.LAGS)
    # A q in [25, maxstart-25] supports all forty negatives, no repeat padding.
    for q in range(25, 50):
        assert all(0 <= q + lag <= 74 for lag in M.LAGS)
        assert all((q + lag) * 640 + 3440 <= 51200 for lag in M.LAGS)
    assert M.energy_bin(np.zeros(3200, np.int16), 0) == 0
    assert M.energy_bin(np.full(3200, 16384, np.int16), 0) == 7
    with pytest.raises(ValueError, match="not complete"):
        M.energy_bin(np.zeros(3199, np.int16), 0)


def test_auc_handles_ties_and_gate_does_not_reward_constant_output():
    assert M.auc_from_scores([1, 2], [0, 0]) == 1
    assert M.auc_from_scores([1, 1], [1, 1]) == .5
    curves = np.ones((4, 21))
    r = M.summarize_validation(np.ones(4), np.ones(40), curves, np.arange(-10, 11))
    assert r["auc"] == .5 and r["gate_pass"] is False
    curves[:, 10] = 0
    r = M.summarize_validation(np.zeros(4), np.ones(40), curves, np.arange(-10, 11))
    assert r["gate_pass"] is True and r["absolute_peak_mae_ms"] == 0


def test_prepare_pairs_rejects_missing_phonetic_supervision(tmp_path):
    # Do not accidentally substitute unsupervised acoustic clusters for phones.
    with pytest.raises(ValueError, match="forced-aligned"):
        M.phone_spans({"id": "x"})


def test_contrastive_weight_changes_only_negative_term():
    torch = pytest.importorskip("torch")
    v = torch.tensor([[0., 0.], [0., 0.]])
    ap = torch.tensor([[1., 0.], [1., 0.]])
    an = torch.tensor([[2., 0.], [3., 0.]])
    loss0, p0, n0 = M.contrastive_loss(ap, an, v, torch.ones(2), 4)
    loss1, p1, n1 = M.contrastive_loss(ap, an, v, torch.tensor([1.5, .5]), 4)
    assert torch.equal(p0, p1) and torch.equal(n0, n1)
    expected = .5 * (((torch.tensor([1.5, .5]) - 1) * torch.relu(4 - n0).square()).mean())
    assert float(loss1 - loss0) == pytest.approx(float(expected), abs=1e-6)


def test_audio_window_cache_never_decodes_video(tmp_path):
    path = tmp_path / "mfcc.npy"
    x = np.arange(13 * 100, dtype=np.float32).reshape(13, 100)
    np.save(path, x)
    cache = M.ClipCache([{"mfcc_path": str(path), "video_path": "nonexistent-video"}])
    assert np.array_equal(cache.audio_window(0, 5), x[:, 20:40][None])
    assert not cache.cache and len(cache.audio_cache) == 1
    with pytest.raises(ValueError, match="incomplete"):
        cache.audio_window(0, -1)


def test_real_pcm_shift_reencodes_official_frontend(tmp_path):
    pytest.importorskip("python_speech_features")
    from scipy.io import wavfile
    import python_speech_features as psf
    rng = np.random.default_rng(13)
    pcm = rng.integers(-6000, 6000, 32000, dtype=np.int16)
    path = tmp_path / "canonical.wav"
    wavfile.write(path, 16000, pcm)
    original = psf.mfcc(pcm, 16000).T.astype(np.float32)
    shifted = M.shifted_mfccs(path)
    assert not np.array_equal(shifted[4], original)
    assert np.allclose(shifted[4][:, 20:100], original[:, 4:84], atol=1e-5)
    assert np.allclose(shifted[-4][:, 4:84], original[:, 20:100], atol=1e-5)


def test_official_architecture_fresh_both_encoders_and_finite_backward():
    torch = pytest.importorskip("torch")
    source = SOURCE.parents[2] / "third_party/syncnet_python/SyncNetModel.py"
    if not source.exists():
        pytest.skip("official model source not installed on this host")
    torch.set_num_threads(2)
    model = M.load_model(source, seed=1729)
    initial = M.state_digest(model.state_dict())
    second = M.load_model(source, seed=1729)
    assert initial == M.state_digest(second.state_dict())
    del second
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            assert torch.count_nonzero(module.running_mean) == 0
            assert torch.equal(module.running_var, torch.ones_like(module.running_var))
            assert int(module.num_batches_tracked) == 0
    assert all(p.requires_grad for p in model.parameters())
    model.train()
    a = model.forward_aud(torch.randn(4, 1, 13, 20))
    v = model.forward_lip(torch.rand(2, 3, 5, 224, 224) * 255)
    assert a.shape == (4, 1024) and v.shape == (2, 1024)
    loss, _, _ = M.contrastive_loss(a[:2], a[2:], v, torch.ones(2), 10)
    loss.backward()
    assert torch.isfinite(loss)
    for prefix in ("netcnnaud", "netcnnlip", "netfcaud", "netfclip"):
        grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_preparation_keeps_one_pool_binds_inputs_and_preserves_holdout(tmp_path):
    from scipy.io import wavfile
    mfcc = tmp_path / "mfcc.npy"
    np.save(mfcc, np.zeros((13, 1200), np.float32))
    audio = tmp_path / "pcm.wav"
    wavfile.write(audio, 16000, np.full(192000, 8000, np.int16))
    base = dict(video_path="unused-in-pair-preparation.mp4", mfcc_path=str(mfcc),
                audio_path=str(audio), frame_count=300)
    rows = []
    rng = np.random.default_rng(23)
    for i in range(24):
        # Repeated tokens plus independent phone transitions give both candidate
        # categories in many signed-lag strata; all audio energy/VAD is matched.
        tokens = rng.choice(["a", "b"], size=60).tolist()
        spans = [[j * .2, (j + 1) * .2, token] for j, token in enumerate(tokens)]
        rows.append(dict(base, id=f"t{i}", speaker=f"t{i}", split="train", phone_spans=spans))
    rows.append(dict(base, id="v0", speaker="heldout", split="val"))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "pool"
    M.prepare_pairs(path, out)
    audit = M.read_json(out / "weight_audit.json")
    assert 0 < audit["retained_negatives"] <= audit["candidate_negatives"]
    assert audit["intervention_mean_weight"] == pytest.approx(1, abs=1e-6)
    assert audit["pairs_sha256"] == M.sha(out / "pairs.npz")
    assert audit["clips_sha256"] == M.sha(out / "clips.json")
    assert audit["similar_effective_fraction"] == pytest.approx(.5, abs=1e-6)
    assert audit["retained_both_voiced_fraction"] == 1
    assert audit["quality_gate"]["passed"] is False  # synthetic fixture below real-data gates
    assert audit["quality_gate"]["checks"]["train_speakers"] is False
    assert audit["max_stratum_mass_error"] < 1e-4
    with np.load(out / "pairs.npz") as pairs:
        assert all(rows[int(i)]["split"] == "train" for i in pairs["clip_index"])
        assert set(pairs["similar"].tolist()) == {0, 1}
    val = M.read_json(out / "validation_queries.json")
    assert len(val) == 4 and all(rows[q["clip_index"]]["speaker"] == "heldout" for q in val)
