import numpy as np
from scripts.experiments.tts_chinese_instance_analysis import contrasts
from scripts.experiments import tts_chinese_instance_analysis as analysis
from scripts.experiments.tts_chinese_instance_quality import edit

def test_diagonal_only_matching():
    q={a+b:float(a==b) for a in ('N','Q1','Q2') for b in ('N','Q1','Q2')}
    c=contrasts(q)
    assert c['STT']==1 and c['SNT']==1 and c['SNT-STT']==0

def test_only_domain_matching():
    q={a+b:float((a=='N')==(b=='N')) for a in ('N','Q1','Q2') for b in ('N','Q1','Q2')}
    c=contrasts(q)
    assert c['STT']==0 and c['SNT']==1

def test_constant_distance_shift_does_not_change_contrasts():
    rng=np.random.default_rng(7)
    q={a+b:float(rng.normal()) for a in ('N','Q1','Q2') for b in ('N','Q1','Q2')}
    a,b=contrasts(q),contrasts({k:v+20 for k,v in q.items()})
    for k in ('STT','SNT','SNT-STT'):assert abs(a[k]-b[k])<1e-12

def test_common_three_arm_nodes_and_donors(monkeypatch):
    ph=[(str(i),.4+i*.24,.4+(i+1)*.24) for i in range(18)]
    monkeypatch.setattr(analysis,'phones',lambda _:ph)
    arms=('N','Q1','Q2');r={'id':'fixed','speaker':'s','grids':{a:{'path':a} for a in arms}}
    s=analysis.support(r,arms,{a:140 for a in arms},3)
    assert s['eligible'] and s['n_occurrences']>=5
    for q in s['queries']:
        node=s['nodes'][q['node']]
        assert len(set(node['j'].values()))==1
        assert len(q['donors'])>=5
        assert all(3<=abs(s['nodes'][d]['j']['N']-node['j']['N'])<=15 for d in q['donors'])
        assert all(abs(x)<=.02000001 for x in node['errors'].values())

def test_phone_sequence_mismatch_not_relaxed(monkeypatch):
    monkeypatch.setattr(analysis,'phones',lambda path:[('a' if path=='N' else 'b',0.,2.)])
    arms=('N','Q1','Q2');r={'id':'fixed','speaker':'s','grids':{a:{'path':a} for a in arms}}
    assert analysis.support(r,arms,{a:100 for a in arms},3)['reason']=='phone_sequence_mismatch'

def test_cer_no_number_or_homophone_mapping():
    assert edit('一','1')==1 and edit('的','得')==1
