"""Temporal support, actual waveform shifting, and paired direction bookkeeping."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('cloud_scores',Path(__file__).parents[1]/'scripts/experiments/cloud_mtd_synchformer_20261028.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_native_support_no_stretch():
    assert m.center_start(59,2)==4
    assert m.center_start(125,5)==0
    try:m.center_start(124,5)
    except ValueError:pass
    else:raise AssertionError('must not pad short clips')


def test_pcm_shift_not_circular_or_gain():
    import numpy as np
    x=np.array([1.,2.,3.,4.],dtype=np.float32)
    assert np.array_equal(m.shift_pcm(x,2),[0,0,1,2])
    assert np.array_equal(m.shift_pcm(x,-2),[3,4,0,0])
    assert np.array_equal(m.shift_pcm(x,0),x)


def test_pair_conflicts_ties_and_source_clusters():
    ps=[dict(source='a',N=2,T=3,delta=1,delta_sync_c=-1),dict(source='a',N=2,T=1,delta=-1,delta_sync_c=1),
        dict(source='b',N=2,T=2,delta=0,delta_sync_c=1)]
    out=m.summarize_pairs(ps,'test')
    assert out['pairs']==3 and out['sources']==2 and out['conflict_count']==2
    assert out['tts_wins']==out['natural_wins']==out['ties']==1
    assert out['sync_tts_metric_natural']==out['sync_natural_metric_tts']==1
    assert out['mean_delta']==0 and out['conflict_fraction']==2/3
