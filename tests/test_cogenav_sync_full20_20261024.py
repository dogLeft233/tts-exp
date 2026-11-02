"""Clustered summaries must treat sources, not repeated draws, as units."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/experiments"))
from cogenav_sync_full20_20261024 import interval, summarize, common_qualified_pairs


def test_constant_source_bootstrap():
    r=interval([.25]*20)
    assert r['mean']==.25 and r['ci95']==[.25,.25] and r['sources']==20


def test_source_balance_with_missing_draws():
    pairs=[]
    for source,draws,value in [('a',3,1.),('b',1,-1.)]:
        for family in ['L06-N','L17-N','L17-L06']:
            for draw in range(draws):
                pairs.append(dict(source=source,family=family,draw=draw,score_float32=1.,
                                  **{f:value for f in ['eq1','zero_common','max_score','zero_excess','peak_excess']}))
    r=summarize(pairs)
    assert r['sources']==2 and r['pairs']==12
    assert r['fields']['eq1']['source_balanced_agreement']['mean']==.5
    assert r['families']['L06-N']['eq1']['mean']==0


def test_qualification_removes_counterpart_draw():
    planned=[dict(sample_id='s1',arm='L06',draw=3,qualified=False),
             dict(sample_id='s1',arm='L17',draw=3,qualified=True)]
    pairs=[dict(source='s1',draw=d,qualified=True,family='L17-N') for d in [1,2,3]]
    assert [p['draw'] for p in common_qualified_pairs(pairs,planned)]==[1,2]
