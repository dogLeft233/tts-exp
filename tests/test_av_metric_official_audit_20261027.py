"""Artifact-backed integration checks for the official-inference audit.

The separately produced audit artifacts are the fixture; no neural forward or
API call occurs here. Missing research fixtures skip in an ordinary checkout.
"""
import json
from pathlib import Path
import unittest

OUT = Path(__file__).resolve().parents[1] / 'runs/av_metric_official_audit_20261027_attempt01'


def read(name):
    path = OUT / name
    if not path.exists():
        raise unittest.SkipTest('Research audit fixture unavailable: ' + name)
    return json.loads(path.read_text(encoding='utf8'))


def test_all_real_control_media_preserve_video_and_exact_audio_shift():
    report = read('media_and_cached_features_audit.json')
    assert len(report['rows']) == 40
    for row in report['rows']:
        assert row['waveform_exact'] and row['video_frames_exact']
        assert row['stream_start_and_rates_equal']
        assert row['peavs_video_features_exact'] and row['peavs_audio_features_changed']


def test_official_sync_pipeline_reproduces_the_counterexample():
    report = read('syncreward_official_parity.json')
    assert len(report['parity']) == 3
    assert max(r['absolute_error'] for r in report['parity']) < 5e-6
    assert report['parity'][1]['official_fp32'] > report['parity'][0]['official_fp32']
    assert report['decoder_resampled_shift_max_error'] < 1e-6
    assert max(r['error'] for r in report['encode_batch1_vs2']) < 1e-4


def test_upstream_peavs_padding_dependence_is_not_hidden_by_the_wrapper():
    report = read('peavs_official_parity.json')
    minimal = read('peavs_minimal_batch_dependency.json')
    assert report['max_cached_head_parity_error'] < 1e-5
    assert report['correlation_cuda_vs_torch_max_error'] < 1e-6
    assert report['max_batch1_vs_default64_error'] > .1
    assert minimal['audio_shape'][0] != minimal['video_shape'][0]
    assert minimal['batch_dependency'] > .1
    assert abs(minimal['equalized_diagnostic_alone'] - minimal['equalized_diagnostic_with_companion']) < 1e-5
    assert minimal['diagnostic_not_official_score']


def test_bf16_controls_are_complete_and_not_silently_fp32():
    report = read('syncreward_bf16_controls.json')
    assert report['controls'] == len(report['rows']) == 40
    assert report['bf16_including_emulation']
    assert 'BF16' in report['precision']
    assert report['lower'] == sum(r['delta'] < 0 for r in report['rows'])


def test_versioned_wrappers_use_the_verified_precision_and_batch_defaults():
    peavs = read('peavs_new_wrapper_smoke.json')
    assert peavs['metric_batch_size'] == 64
    batch = read('peavs_native_default64_reanalysis.json')
    source = next(r for r in batch['pairs'] if r['source'] == 'BAC009S0008W0121')
    observed = next(r['score'] for r in peavs['scores'] if r['filename'] == 'native__BAC009S0008W0121_N_0')
    assert abs(observed-source['natural']) < 1e-5
    path = OUT / 'syncreward_new_wrapper_smoke.jsonl'
    if not path.exists():
        raise unittest.SkipTest('Versioned SyncReward smoke is pending')
    row = json.loads(path.read_text().splitlines()[0])
    assert row['precision'] == 'bf16' and row['encode_batch_size'] == 1
    control = next(r for r in read('syncreward_bf16_controls.json')['rows'] if r['id'].startswith('shift__BAC009S0002W0122__'))
    assert row['reward'] == control['native']
