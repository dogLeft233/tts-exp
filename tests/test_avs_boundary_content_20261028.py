"""Actual artifact checks for target-preserving AVS boundary manipulations."""
import json
from pathlib import Path

import numpy as np
import pytest

RUN=Path(__file__).resolve().parents[1]/'runs/avs_boundary_content_cloud_20261028_attempt01'


def read(name):
    if not (RUN/name).exists():pytest.skip('Actual boundary experiment artifacts absent')
    return json.loads((RUN/name).read_text())


def cos(a,v):
    a,v=a.astype(float),v.astype(float)
    return np.sum(a*v,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(v,axis=1))


def test_full_cohort_and_native_reproduction():
    rows=read('items.json')
    assert len(rows)==80 and len({r['source'] for r in rows})==20
    assert sum(r['arm']=='N' for r in rows)==20
    assert max(max(r['native_parity'].values()) for r in rows)<=1e-5
    assert all(all(x['audio_exact'] and x['video_exact'] for x in r['target_input_equality']) for r in rows)


def test_actual_context_mappings_keep_original_and_target_data():
    for r in read('items.json'):
        with np.load(RUN/'items'/f"{r['id']}.npz") as z:
            f=r['frames'];t=(f-5)//2
            assert np.array_equal(z['target_indices'],np.arange(t,t+5))
            assert np.array_equal(z['reflect_map'],np.pad(np.arange(f),(75,75),mode='reflect'))
            assert np.array_equal(z['repeat_map'],np.pad(np.arange(f),(75,75),mode='edge'))
            for mapping in [z['reflect_map'],z['repeat_map']]:
                assert np.array_equal(mapping[75:75+f],np.arange(f))
                assert np.array_equal(z['input_audio'][mapping][75:75+f],z['input_audio'])
                assert np.array_equal(z['input_video'][mapping][75:75+f],z['input_video'])
            for start,local in zip(z['crop_starts'],z['target_local_starts']):
                assert 0<=start and start+25<=f
                assert np.array_equal(z['input_audio'][start:start+25][local:local+5],z['input_audio'][t:t+5])
                assert np.array_equal(z['input_video'][start:start+25][local:local+5],z['input_video'][t:t+5])


def test_visual_proxy_and_native_frame_support():
    for r in read('items.json'):
        with np.load(RUN/'items'/f"{r['id']}.npz") as z:
            valid=z['lip_valid'];p=z['lip_points']
            ratio=np.linalg.norm(p[:,0]-p[:,1],axis=1)/np.linalg.norm(p[:,2]-p[:,3],axis=1)
            assert np.allclose(ratio[valid],z['lip_ratio'][valid],rtol=0,atol=1e-7)
            assert np.isnan(z['lip_ratio'][~valid]).all()
            assert valid.sum()==r['valid_landmark_frames']
            assert np.isfinite(z['frame_rms_ratio']).all()
            assert z['input_audio'].shape[0]==z['input_video'].shape[0]==r['frames']


def test_all_reported_regional_and_target_scores_from_saved_embeddings():
    for r in read('items.json'):
        with np.load(RUN/'items'/f"{r['id']}.npz") as z:
            for c in ['native','reflect','repeat']:
                scores=cos(z[c+'_audio'],z[c+'_video'])
                for name,ix in zip(['head','center','tail'],z['segment_indices']):
                    actual=float(scores[ix+(75 if c!='native' else 0)].mean())
                    assert abs(actual-r['regional_scores'][c][name])<1e-12
            for c,start in zip(['crop_head','crop_mid','crop_tail'],[0,10,20]):
                actual=float(cos(z[c+'_audio'],z[c+'_video'])[start:start+5].mean())
                assert abs(actual-r['target_scores'][c])<1e-12


def test_source_cluster_bootstrap_and_state_coverage():
    rows=read('analysis_rows.json');summary=read('summary.json')
    for view,subset in [('all20',rows),('strict12',[r for r in rows if r['strict_reference']])]:
        for arm in ['N','C']:
            rr=[r for r in subset if r['arm']==arm];sources=sorted({r['source'] for r in rr})
            assert summary[view][arm]['sources']==len(sources)
            for field in rr[0]['values']:
                values=np.array([np.mean([r['values'][field] for r in rr if r['source']==s]) for s in sources])
                ix=np.random.default_rng(20261028).integers(0,len(values),(10000,len(values)))
                old=summary[view][arm]['effects'][field]
                assert abs(values.mean()-old['mean'])<1e-12
                assert np.allclose(np.quantile(values[ix].mean(1),[.025,.975]),old['ci95'],rtol=0,atol=1e-12)
            for setting,states in summary[view][arm]['state_edge_minus_center'].items():
                for state,data in states.items():
                    assert data['sources']==sum(n>0 for n in data['matched_clip_counts'].values())
