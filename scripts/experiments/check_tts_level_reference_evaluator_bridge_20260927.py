"""Independent NumPy distance, E difference, support and speaker-bootstrap audit."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_reference_evaluator_bridge_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def distance(v,a):
    n=len(v);result=np.empty((n,31));v=np.asarray(v,dtype=np.float32);a=np.asarray(a,dtype=np.float32)
    for j,lag in enumerate(range(-15,16)):
        partner=np.zeros_like(a);idx=np.arange(n);ok=(idx+lag>=0)&(idx+lag<n);partner[ok]=a[idx[ok]+lag];delta=(v-partner+np.float32(1e-6)).astype(float);result[:,j]=np.sqrt(np.einsum('ij,ij->i',delta,delta))
    return result

def stat(vals,groups):
    names=sorted(set(groups));m=np.array([sum(float(v) for v,g in zip(vals,groups) if g==n)/sum(g==n for g in groups) for n in names]);indices=np.random.Generator(np.random.PCG64(20260926)).integers(len(names),size=(20000,len(names)));samples=np.sum(m[indices],axis=1)/len(names)
    return float(sum(m)/len(m)),np.percentile(samples,[.5,99.5])

def main():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip();iv=read(OUT/'input_validation.json');lock=read(OUT/'score_lock.json');assert sha(OUT/'input_validation.json')==lock['input_validation_sha256']
    for f,h in {**p['bindings'],**iv['all_inputs'],**lock['code']}.items():assert sha(f)==h,f
    de=0.;me=0.;entries=0
    for row in p['rows']:
        saved=read(OUT/'scores'/(row['id']+'.json'));mp=OUT/'scores'/(row['id']+'.npz');assert sha(mp)==saved['matrix_sha256'];mats=np.load(mp)
        for key,cells in saved['cells'].items():
            kind,geom,a,c,ac=key.split('/');r=row['arms'][a];n=r['L'];vp=r['source_visual'][c] if kind=='generated' else p['source_only'][c];v=np.load(vp)['visual'][:n];audio=np.load(r['level_features'][ac])['audio'][:n]
            if geom=='unit':v=(v.astype(float)/np.sqrt(np.sum(v.astype(float)**2,axis=1))[:,None]).astype(np.float32);audio=(audio.astype(float)/np.sqrt(np.sum(audio.astype(float)**2,axis=1))[:,None]).astype(np.float32)
            m=distance(v,audio);de=max(de,float(abs(m-mats[key]).max()));entries+=m.size
            for policy,s in cells.items():
                curve=m[20:n-20].mean(0) if policy=='guard20' else m.mean(0) if policy=='guard0' else np.array([m[max(0,-lag):min(n,n-lag),lag+15].mean() for lag in range(-15,16)])
                B=float(np.median(curve));D=float(min(curve));expected={'C':B-D,'B':B,'D':D,'C_anchor':B-curve[18],'D_anchor':curve[18]};me=max(me,max(abs(expected[k]-s[k]) for k in expected))
        print('checked',row['id'],flush=True)
    assert de<1e-5 and me<1e-5
    summary=read(OUT/'summary.json');effects=read(OUT/'effects.json');groups={};err=0.
    for r in effects:
        assert abs(r['E']-(r['LEVEL']-r['raw']))<1e-12
        key='/'.join([r['subset'],r['kind'],r['geometry'],r['policy']]),r['condition'],r['metric'];groups.setdefault(key,{}).setdefault(r['id'],{})[r['arm']]=r
    for (view,c,metric),data in groups.items():
        ids={r['id'] for r in p['rows'] if view.startswith('all74') or r['split']=='evaluation'};assert set(data)==ids;records=list(data.values());speakers=[r['N']['speaker'] for r in records]
        vals={'N':[r['N']['E'] for r in records],'T':[r['T']['E'] for r in records],'T_minus_N':[r['T']['E']-r['N']['E'] for r in records],'raw_gap':[r['T']['raw']-r['N']['raw'] for r in records],'LEVEL_evaluation_gap':[r['T']['LEVEL']-r['N']['LEVEL'] for r in records]}
        for name,values in vals.items():
            mean,ci=stat(values,speakers);s=summary[view][c][metric][name];err=max(err,abs(mean-s['mean']),float(abs(ci-s['ci99']).max()));assert s['n']==len(ids)
    assert err<1e-10
    result={'passed':True,'distance_entries':entries,'max_distance_error':de,'max_metric_error':me,'max_statistics_error':err,'exact_cohorts':True,'checker_sha256':sha(__file__)};(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
