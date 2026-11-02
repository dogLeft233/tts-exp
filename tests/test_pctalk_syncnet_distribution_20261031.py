from pathlib import Path
import importlib.util
S=Path(__file__).resolve().parents[1]/'scripts/analysis/pctalk_syncnet_distribution_20261031.py'
spec=importlib.util.spec_from_file_location('pc_dist',S);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_source_average_not_best_draw_and_duplicates_not_extra_sources():
    rows=[dict(id='n',source='a',speaker='s',strict_reference=True,arm='N',SyncNet=3.)]
    rows += [dict(id='t'+str(i),source='a',speaker='s',strict_reference=True,arm='T',SyncNet=x,audio_sha256=str(x),video_sha256=str(x)) for i,x in enumerate([4.,4.,7.])]
    pairs,draws=m.pair_rows(rows)
    assert len(pairs)==1 and len(draws)==3 and pairs[0]['delta']==2.

def test_histogram_boundary_and_sample_standard_deviation():
    r=m.distribution([-1.,0.,.5,1.,2.])
    assert [x['count'] for x in r['bins']]==[0,1,1,1,1,1]
    assert r['positive']==3 and r['negative']==1 and r['ties']==1
    assert abs(r['sample_sd']-(5/4)**.5)<1e-12
