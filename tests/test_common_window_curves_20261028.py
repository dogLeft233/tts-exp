"""Artifact-level checks for actual common-window curves, no synthetic model mocks."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'runs/common_window_curves_cloud_20261028_attempt01'


def load(name):
    if not (ROOT/name).exists():
        pytest.skip('actual experiment artifacts unavailable')
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def test_actual_native_support_and_equal_source_counts():
    rows=load('inputs.json')
    assert len(rows)==80
    for source in {x['sample_id'] for x in rows}:
        group=[x for x in rows if x['sample_id']==source]
        assert len(group)==4 and len({x['query_count'] for x in group})==1
    for r in rows:
        q0,q1=r['query_start'],r['query_end_exclusive']
        assert q0>=15 and q1+15<=r['sync_windows']
        assert r['sync_windows']+5<=r['common_native_frames']
        assert r['common_native_frames']<=min(r['source_frames'],r['samples']//640)


def test_actual_curve_reductions_and_decomposition():
    items=load('items.json')
    pairs=load('pairs.json')
    assert len(items)==80 and len(pairs)==60
    for row in items:
        with np.load(ROOT/'items'/f"{row['id']}.npz") as z:
            for m in ['sync','sync_unit','avs','co','co_raw']:
                c=z[m+'_matrix'].mean(0)
                e=row['endpoints'][m]
                assert np.allclose(c,e['curve'],rtol=0,atol=1e-12)
                assert abs(float(c.max()-np.median(c))-e['contrast'])<1e-12
    for p in pairs:
        for d in p['delta'].values():
            assert abs(d['contrast']-d['best_contribution']-d['background_contribution'])<1e-12


def test_canonical_audio_and_original_feature_parity():
    rows=load('inputs.json')
    parity=load('parity.json')
    assert parity['status']=='PASS' and len(parity['fixtures'])==3
    assert max(x['official_audio_feature_maxerror'] for x in parity['fixtures'])<=1e-5
    assert all(x['historical_clock']['lag_samples_scoring_minus_canonical']==1105 for x in rows if x['arm']=='N')


def test_controls_report_actual_relative_peak_errors():
    c=load('controls.json')
    expected={'sync':6,'avs':40,'co':160}
    for metric,total in expected.items():
        rows=[x for x in c['rows'] if x['metric']==metric]
        assert len(rows)==total
        assert c['summary'][metric]['passed']==sum(abs(x['relative_error_frames'])<=1 for x in rows)
        assert c['summary'][metric]['status']==('PASS' if c['summary'][metric]['passed']/total>=.8 else 'CONTROL_FAILED')


def test_upstream_cache_input_identity_binding():
    binding=load('provenance_binding.json')
    assert binding['status']=='PASS' and binding['count']==80
    assert all(x['audio_match'] and x['video_match'] for x in binding['rows'])
