"""Integration checks against real SparseSync artifacts; no simulated model scores."""
import json
from pathlib import Path

import pytest


RUN = Path(__file__).parents[1]/'runs/sparsesync_official_cloud_rescore_20261028_attempt01'


def artifact(name):
    path=RUN/name
    if not path.exists():
        pytest.skip('Real deployment artifact not yet available: '+name)
    return json.loads(path.read_text(encoding='utf-8'))


def test_real_official_example_parity_and_strict_model_load():
    p=artifact('official_parity.json')
    assert p['status']=='PASS' and p['official_main_executed']
    assert p['audio_max_abs_diff']==p['video_max_abs_diff']==0
    assert p['logits_max_abs_diff']<=1e-5
    assert p['probabilities_max_abs_diff']<=1e-6
    m=artifact('model_load.json')
    assert m['strict'] and m['full_checkpoint_bytes']==653037857
    assert len(m['grid'])==21


def test_real_native_support_and_pair_denominators():
    rows=artifact('inputs.json');scores=artifact('scores.json');pairs=artifact('paired_scores.json')
    eligible={r['id'] for r in rows if r['eligible_5s']}
    assert set(scores)==eligible
    assert all(s['pcm_exact'] for s in scores.values())
    nmap={r['sample_id']:r['id'] for r in rows if r['arm']=='N'}
    expected={p['id'] for p in artifact('sync_pairs.json') if p['id'] in eligible and nmap[p['source']] in eligible}
    for ps in pairs.values():
        assert {p['id'] for p in ps}==expected
        assert len(ps)==len(expected)


def test_real_physical_shift_controls_and_audit():
    c=artifact('known_shift_controls.json')
    eligible_n=sum(r['arm']=='N' and r['eligible_5s'] for r in artifact('inputs.json'))
    assert c['total']==2*eligible_n
    for row in c['rows']:
        assert abs(row['shift_samples'])==6400
        assert row['expected_relative_offset']==-row['shift_samples']/16000
        assert row['p_zero_lower']==(row['shifted']['p_zero']<row['native']['p_zero'])
    audit=artifact('independent_audit.json')
    assert audit['status']=='PASS' and audit['softmax_max_abs_diff']<2e-6


def test_real_saturation_diagnostic_preserves_precision_check():
    d=artifact('saturation_precision_diagnostic.json')
    assert d['posthoc_diagnostic'] and d['float64_sign_agree']==d['pairs']==18
    scores=artifact('scores.json')
    assert len(scores)==28
    assert all(r['output']['p_zero']>.9995 for r in scores.values())
    assert all(r['output']['predicted_offset']==0 for r in scores.values())
