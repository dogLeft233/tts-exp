"""Independent NumPy distance, four-cell contrast, and bootstrap audit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[2]/'runs/tts_acoustic_generation_cross_20260926'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def independent_matrix(v,a):
    # Explicit per-lag valid and zero-padded audio, float64 norm of float32 eps offset.
    v=np.asarray(v,dtype=np.float32);a=np.asarray(a,dtype=np.float32);n=len(v);out=np.empty((n,31))
    for column,lag in enumerate(range(-15,16)):
        partner=np.zeros_like(a);i=np.arange(n);valid=(i+lag>=0)&(i+lag<n);partner[valid]=a[i[valid]+lag]
        difference=(v-partner+np.float32(1e-6)).astype(np.float64)
        out[:,column]=np.sqrt(np.einsum('ij,ij->i',difference,difference))
    return out

def independent_stats(values,groups):
    names=sorted(set(groups));means=[]
    for name in names:means.append(sum(float(v) for v,g in zip(values,groups) if g==name)/sum(g==name for g in groups))
    means=np.array(means);rng=np.random.Generator(np.random.PCG64(20260926));draws=rng.integers(0,len(names),(20000,len(names)));boot=np.sum(means[draws],axis=1)/len(names)
    return float(sum(means)/len(means)),np.percentile(boot,[.5,99.5])

def check(split='all'):
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip();support=read(OUT/'support.json');rows=[r for r in support if split=='all' or r['split']==split]
    checked=0;maximum_distance=0.;maximum_metric=0.;closure=0.;metric_count=0
    for row in rows:
        sid=row['id'];scores=read(OUT/'scores'/(sid+'.json'));mp=OUT/'scores'/(sid+'.npz');assert sha(mp)==scores['matrix_sha256'];saved=np.load(mp)
        for arm in ['N','T']:
            L=row['arms'][arm]['joint_L'];features={c:np.load(OUT/'features'/sid/arm/(c+'.npz')) for c in p['conditions']}
            for geom in ['raw','unit']:
                for key,cell in scores['cells'][arm][geom].items():
                    if key=='historical':continue
                    vc,ac=key.split('__');v=features[vc]['visual'][:L];a=features[ac]['audio'][:L]
                    if geom=='unit':v=(v.astype(float)/np.sqrt(np.sum(v.astype(float)**2,axis=1))[:,None]).astype(np.float32);a=(a.astype(float)/np.sqrt(np.sum(a.astype(float)**2,axis=1))[:,None]).astype(np.float32)
                    expected=independent_matrix(v,a);actual=saved[arm+'__'+geom+'__'+key];err=float(abs(expected-actual).max());maximum_distance=max(maximum_distance,err);assert err<1e-5;checked+=expected.size
                    for policy,metrics in cell['policies'].items():
                        if metrics is None:assert L<=40 and policy=='guard20';continue
                        if policy=='guard20':curve=expected[20:L-20].mean(0)
                        elif policy=='guard0':curve=expected.mean(0)
                        else:curve=np.array([expected[max(0,-lag):min(L,L-lag),lag+15].mean() for lag in range(-15,16)])
                        B=float(np.median(curve));D=float(curve.min());values={'C':B-D,'B':B,'D':D,'C_anchor':B-float(curve[18]),'D_anchor':float(curve[18])}
                        maximum_metric=max(maximum_metric,max(abs(value-metrics[k]) for k,value in values.items()));metric_count+=len(values)
        print('checked',sid,flush=True)
    stats_error=0.;summary=None
    if split=='all' and (OUT/'summary.json').exists():
        effects=read(OUT/'effects.json');summary=read(OUT/'summary.json');groups={}
        for r in effects:
            G=r['q10']-r['q00'];E=r['q01']-r['q00'];I=r['q11']-r['q10']-r['q01']+r['q00'];total=r['q11']-r['q00'];closure=max(closure,abs(G+E+I-total))
            assert max(abs(r[k]-v) for k,v in [('G',G),('E',E),('I',I),('total',total)])<1e-12
            key=r['geometry']+'/'+r['policy'],r['metric'],r['condition'];groups.setdefault(key,{}).setdefault(r['id'],{})[r['arm']]=r
        for (view,metric,condition),data in groups.items():
            pairs=list(data.values());speakers=[v['N']['speaker'] for v in pairs]
            for effect in ['G','E','I','total']:
                for arm in ['N','T','T_minus_N']:
                    vals=[d[arm][effect] if arm!='T_minus_N' else d['T'][effect]-d['N'][effect] for d in pairs];mean,ci=independent_stats(vals,speakers);expected=summary['results'][view][metric]['conditions'][condition][effect][arm]
                    stats_error=max(stats_error,abs(mean-expected['mean']),float(max(abs(ci-expected['ci99']))))
        for view, block in summary['results'].items():
            for metric, detail in block.items():
                for dose in ['lo','hi']:
                    rgroup=groups[view,metric,'R'+dose];egroup=groups[view,metric,'ENV'+dose]
                    ids=list(rgroup);speakers=[rgroup[sid]['N']['speaker'] for sid in ids]
                    for effect in ['G','E','I','total']:
                        vals_by_arm={arm:[rgroup[sid][arm][effect]-egroup[sid][arm][effect] for sid in ids] for arm in ['N','T']}
                        vals_by_arm['T_minus_N']=[t-n for t,n in zip(vals_by_arm['T'],vals_by_arm['N'])]
                        for arm,values in vals_by_arm.items():
                            mean,ci=independent_stats(values,speakers);expected=detail['R_minus_ENV'][dose][effect][arm]
                            stats_error=max(stats_error,abs(mean-expected['mean']),float(max(abs(ci-expected['ci99']))))
        assert stats_error<1e-10
    result={'status':'passed','split':split,'distance_entries':checked,'metric_entries':metric_count,'max_distance_error':maximum_distance,'max_metric_error':maximum_metric,'max_fourcell_closure':closure,'max_statistics_error':stats_error,'checker_sha256':sha(__file__)}
    (OUT/('independent_validation_'+split+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['all','calibration','evaluation'],default='all');check(parser.parse_args().split)
