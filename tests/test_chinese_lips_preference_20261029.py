"""Controlled-scorer endpoints, support and paired cluster inference contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "chinese_lips_preference", ROOT / "scripts/experiments/chinese_lips_preference_20261029.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_pairwise_eps_orientation_and_physical_zero():
    audio = np.zeros((50, 1024), np.float32)
    video = np.zeros_like(audio)
    audio[:, 0] = np.arange(50)
    video[:, 0] = np.arange(50)
    queries = np.arange(15, 35)
    matrix = MOD.score_matrix(audio, video, queries)
    assert matrix.shape == (20, 31)
    assert np.allclose(matrix[:, 15], -np.sqrt(1024)*1e-6, rtol=1e-12)
    assert np.allclose(matrix[:, 16], -np.sqrt((-1+1e-6)**2+1023e-12), rtol=1e-12)
    endpoints = MOD.reduce_curve(matrix)
    assert endpoints["peak_lag_frames"] == 0
    assert endpoints["zero"] == endpoints["best"]
    assert endpoints["search_uplift"] == 0
    assert endpoints["contrast"] == endpoints["best"] - endpoints["background"]


def test_lag_support_cannot_wrap_or_pad():
    x = np.ones((40, 1024), np.float32)
    for queries in ([14], [25], []):
        with pytest.raises(ValueError, match="support"):
            MOD.score_matrix(x, x, queries)


@pytest.mark.parametrize("payload,expected", [
    ({}, False), ({"status": "COMPLETE"}, False),
    ({"gate_pass": False, "auc": .99}, False),
    ({"gate_pass": True}, True), ({"gate_pass": "true"}, False),
    ({"gate_pass": True, "status": "FAIL"}, False),
    ({"status": "PASS", "passed": False}, False),
    ({"gate": {"status": "PASS", "passed": True}}, True),
])
def test_capability_gate_fails_closed(payload, expected):
    assert MOD.gate_passes(payload) is expected


def test_aishell_source_speaker_is_not_reference_face():
    assert MOD.source_speaker({"sample_id": "BAC009S0002W0122"}) == ("S0002", "AISHELL_source_id_Sdddd")
    assert MOD.source_speaker({"sample_id": "other"}) == ("other", "source_only_unknown_speaker")
    assert MOD.source_speaker({"sample_id": "other", "speaker_id": "S7"})[0] == "S7"


def test_cluster_bootstrap_keeps_same_speaker_sources_together():
    values = {"a": 0.0, "b": 0.2, "c": 1.0}
    clusters = {"a": "speakerA", "b": "speakerA", "c": "speakerB"}
    result = MOD.bootstrap_source_equal(values, clusters, replicates=10000, seed=41)
    assert result["clusters"] == 2 and result["sources"] == 3
    assert result["mean"] == pytest.approx(.4)
    rng = np.random.default_rng(41)
    ix = rng.integers(0, 2, (10000, 2))
    manual = np.array([.2, 1])[ix].sum(1) / np.array([2, 1])[ix].sum(1)
    assert result["ci95"] == pytest.approx(np.quantile(manual, [.025, .975]))


def pair(key, source, speaker, delta, strict=True):
    return dict(id=key, source=source, speaker=speaker,
                speaker_provenance="test", strict_reference=strict,
                delta={field: delta for field in MOD.ENDPOINTS})


def test_cross_condition_uses_paired_wins_not_incommensurate_units():
    # N1 uses a 100x distance scale. Within-model direction is all that this
    # cross-condition comparison uses, so the common positive pair stays tied.
    n0 = [pair("a1", "a", "s1", .01), pair("a2", "a", "s1", -.01),
          pair("b1", "b", "s2", -.02, False)]
    n1 = [pair("a1", "a", "s1", 1), pair("a2", "a", "s1", 2),
          pair("b1", "b", "s2", -3, False)]
    result = MOD.compare_models({"N0": n0, "N1": n1}, replicates=1000)
    assert result["summary"]["all20"]["zero_winrate_N1_minus_N0"]["mean"] == pytest.approx(.25)
    assert result["summary"]["all20"]["changed_tts_win_status"] == 1
    assert result["summary"]["all20"]["no_cross_model_distance_subtraction"] is True
    with pytest.raises(ValueError, match="different"):
        MOD.compare_models({"N0": n0, "N1": n1[:-1]})


def test_source_equal_means_dont_treat_tts_draws_as_independent_sources():
    pairs = [pair("a1", "a", "speakerA", 2), pair("a2", "a", "speakerA", 4),
             pair("b1", "b", "speakerB", -1, False)]
    summary = MOD.summarize_pairs(pairs, replicates=1000)
    zero = summary["metrics"]["zero"]
    assert zero["score_delta_in_this_model_units"]["mean"] == 1
    assert zero["tts_winrate"]["mean"] == .5
    assert zero["tts_higher"] == 2 and zero["natural_higher"] == 1
    assert summary["sources"] == 2 and summary["pairs"] == 3


def test_paired_training_identity_cannot_hide_different_initialization():
    common = dict(seed=1729, initial_model_state_sha256="init", model_source_sha256="source",
                  trainer_source_sha256="trainer", pairs_sha256="pairs", pool_protocol_sha256="pool", batch_size=16,
                  lr=.0001, margin=3, steps=5000, grad_clip=10, val_every=500,
                  val_queries=128, selection="gate then AUC then MAE")
    checkpoints = {"N0": dict(common, condition="N0"), "N1": dict(common, condition="N1")}
    assert MOD.paired_training_identity(checkpoints) == common
    checkpoints["N1"]["initial_model_state_sha256"] = "different"
    with pytest.raises(ValueError, match="initial_model_state"):
        MOD.paired_training_identity(checkpoints)


def test_actual_frozen_cohort_common_support_and_speaker_counts():
    path = ROOT / "runs/common_window_curves_cloud_20261028_attempt01/inputs.json"
    if not path.exists():
        pytest.skip("historical cohort unavailable on this checkout")
    rows = json.loads(path.read_text(encoding="utf-8"))
    groups = MOD.validate_inputs(rows)
    assert len(rows) == 80 and len(groups) == 20
    assert sum(r["arm"] == "N" and r["strict_reference"] for r in rows) == 12
    assert {MOD.source_speaker(r)[1] for r in rows} == {"AISHELL_source_id_Sdddd"}
    assert len({MOD.source_speaker(r)[0] for r in rows}) <= 20
    damaged = [dict(r) for r in rows]
    damaged[0]["query_start"] = 14
    with pytest.raises(ValueError, match="support"):
        MOD.validate_inputs(damaged)


def test_actual_official_frozen_source_reload_forward_parity_cpu(tmp_path):
    source = ROOT / "third_party/syncnet_python/SyncNetModel.py"
    if not source.exists():
        pytest.skip("official source unavailable in this checkout")
    torch = pytest.importorskip("torch")
    frozen = tmp_path / "frozen" / "SyncNetModel.py"
    frozen.parent.mkdir()
    shutil.copy2(source, frozen)
    spec = importlib.util.spec_from_file_location("lips_original_frozen_test", frozen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(1729)
    original = module.S().eval()
    checkpoint = tmp_path / "controlled.pt"
    torch.save(dict(model=original.state_dict(), protocol={
        "model_source": str(frozen.resolve()), "model_source_sha256": MOD.sha(frozen)}), checkpoint)
    # No repo copy exists: loading must use exactly the frozen training source.
    loaded, _ = MOD.load_net(tmp_path / "no_repo", checkpoint, "cpu")
    assert not loaded.training
    assert all(torch.equal(original.state_dict()[k], v) for k, v in loaded.state_dict().items())
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.inference_mode():
            audio = torch.randn(1, 1, 13, 20)
            video = torch.randn(1, 3, 5, 224, 224)
            assert torch.equal(original.forward_aud(audio), loaded.forward_aud(audio))
            assert torch.equal(original.forward_lip(video), loaded.forward_lip(video))
    finally:
        torch.set_num_threads(threads)
    frozen.write_text(frozen.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        MOD.load_net(tmp_path / "no_repo", checkpoint, "cpu")
