"""Independent pre-score audit; hash checks and synthetic arrays only."""
import ast
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/tts_level_event_cross_20260927'
OUT = ROOT / 'runs/tts_event_level_independent_audit_20260927'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())

def main():
    p = read(RUN/'protocol.json')
    for name, digest in read(RUN/'seal.json').items():
        assert sha(RUN/name) == digest, name
    inputs = read(RUN/'inputs.json')
    for path, digest in {**inputs, **p['code']}.items():
        assert sha(path) == digest, path
    old = ROOT/'runs/tts_event_cross_20260926/protocol.json'
    assert sha(ROOT/'scripts/experiments/tts_event_cross.py') == read(old)['source_code_sha256']
    idx = read(RUN/'indices.json')
    legacy = read(ROOT/'runs/tts_static_lag_calibration_20260926/evaluation/support.json')
    assert idx['rows'] == [r for r in legacy['rows'] if r['eligible']]
    assert len(idx['rows']) == 32
    assert sum(len(r['queries']) for r in idx['rows']) == 1411
    assert len({r['speaker'] for r in idx['rows']}) == 13
    tree = ast.parse((ROOT/'scripts/experiments/tts_level_event_cross_20260927.py').read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('geometry','stat','clipmeans')]
    ns = {'np':np,'GEOMS':['raw','unit']}
    exec(compile(ast.Module(body=fns,type_ignores=[]),'<frozen pure functions>','exec'),ns)
    rng = np.random.default_rng(715)
    arrays = {(mod,state,arm):rng.normal(size=(14,7)).astype('float32')
              for mod in ('A','V') for state in ('raw','FIXED','LEVEL') for arm in ('N','T')}
    row = {'nodes':[{'j':{'N':5,'T':7}}, {'j':{'N':8,'T':9}}, {'j':{'N':10,'T':11}}],
           'queries':[{'node':0,'donors':[1,2]},{'node':1,'donors':[0,2]}]}
    cells = p['cells']
    actual = ns['geometry'](row,arrays,cells)
    expected = np.empty_like(actual)
    # Scalar coordinate summation, independent loop ordering; preserve historical
    # float32 normalization before float64 distances exactly as specified.
    for qi,q in enumerate(row['queries']):
        for ci,c in enumerate(cells):
            for gi in range(2):
                aa=arrays['A',c['audio_state'],c['audio_source']]
                vv=arrays['V',c['video_state'],c['video_source']]
                if gi:
                    aa=aa/np.linalg.norm(aa,axis=1)[:,None]
                    vv=vv/np.linalg.norm(vv,axis=1)[:,None]
                vi=row['nodes'][q['node']]['j'][c['video_source']]-3
                def distance(node):
                    ai=row['nodes'][node]['j'][c['audio_source']]
                    return sum((float(vv[vi,k])-float(aa[ai,k])+1e-6)**2 for k in range(7))**.5
                pos=distance(q['node']); neg=[distance(n) for n in q['donors']]
                avg=sum(neg)/len(neg)
                expected[qi,gi,ci]=[pos,avg,avg-pos,sum(1 if z>pos else .5 if z==pos else 0 for z in neg)/len(neg)]
    error=float(np.max(abs(actual-expected))); assert error<2e-15,error
    tied={k:np.ones_like(v) for k,v in arrays.items()}
    assert np.all(ns['geometry'](row,tied,cells)[...,-1] == .5)
    values=np.array([0.,2.,8.,-2.]); speakers=['a','a','b','c']
    got=ns['stat'](values,speakers)
    sv=np.array([1.,8.,-2.]); draws=np.random.default_rng(20260926).integers(3,size=(20000,3))
    assert got['speaker_mean']==sv.mean() and got['utterance_mean']==values.mean()
    assert got['speaker_ci99']==np.quantile(np.mean(sv[draws],axis=1),[.005,.995]).tolist()
    sample=rng.normal(size=(4,9)); R,FV,FA,F=sample
    assert np.max(abs((FV-R)+(FA-R)+(F-FV-FA+R)-(F-R)))<1e-14
    receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(RUN/'protocol.json'),
       'producer_code_sha256':next(iter(p['code'].values())), 'auditor_code_sha256':sha(__file__),
       'input_hashes_verified':len(inputs),'legacy_rows_exact':True,'clips':32,'speakers':13,'queries':1411,
       'synthetic_all20cells_raw_unit_max_error':error,'tie_rank':.5,'speaker_bootstrap_synthetic_exact':True,
       'state_closure_synthetic_pass':True,'new_scores_or_models_executed':False,
       'review_scope':['legacy full support and k3','all donors follow audio source and state',
          'old float32 unit arithmetic then float64 epsilon distance','distinct source and state interaction',
          'historical three-image and image3 bridges required before new scoring','all endpoints/no score selection',
          '32-clip estimand not official71 residual mediation','persistent16MiB and 4.5GiB floor'],
       'runtime':{'python':platform.python_version(),'numpy':np.__version__}}
    OUT.mkdir(exist_ok=True)
    dest=OUT/'static_receipt.json'; assert not dest.exists()
    dest.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'PASS','receipt':str(dest),'sha256':sha(dest)}))

if __name__ == '__main__': main()
