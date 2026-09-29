"""Independent CPU checks; does not import the source-conditioning producer."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_source_conditioning_20260926'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def calculate(m,policy,k):
    n=len(m)
    if policy=='valid':
        c=np.asarray([np.mean(m[np.arange(n)[(np.arange(n)+lag>=0)&(np.arange(n)+lag<n)],lag+15]) for lag in range(-15,16)])
    elif policy=='guard20':c=m[20:-20].mean(axis=0)
    else:c=m.mean(axis=0)
    b=float(np.median(c));d=float(c.min());anchor=float(c[k+15])
    return dict(C=b-d,B=b,D=d,C_anchor=b-anchor,D_anchor=anchor,best_lag=int(np.argmin(c))-15)

def main():
    p=read(OUT/'protocol.json');lock=read(OUT/'score_lock.json');release=read(OUT/'eval_release.json')
    for name,key in [('protocol.json','protocol_sha256'),('support.json','support_sha256'),('geometry.json','geometry_sha256'),('feature_seal.json','feature_seal_sha256')]:assert sha(OUT/name)==lock[key],name
    assert sha(OUT/'calibration.json')==release['calibration_sha256']
    for path,h in p['code_and_models'].items():assert sha(path)==h,path
    for path,h in read(OUT/'feature_seal.json').items():assert sha(path)==h,path
    rows=read(OUT/'support.json');assert len(rows)==74 and sum(r['split']=='evaluation' for r in rows)==48
    geo=read(OUT/'geometry.json');box=np.asarray(geo['generation_boxes']);roi=geo['score_roi'];lower=box[:,:2].min(0);upper=box[:,2:].max(0);side=int(np.ceil(max(upper-lower)*1.5));origin=np.floor((upper+lower-side)/2).astype(int);assert roi==[int(origin[0]),int(origin[1]),int(origin[0]+side),int(origin[1]+side)]
    errors=dict(direct_distances=0.,score_metrics=0.,speaker_mean=0.,ci99=0.,source_index=0.);direct_count=0;allmetrics={};cal_lags=[];conds=('D0','D66','S0','S66')
    for r in rows:
        sid=r['id'];f=OUT/'scores'/r['split']/(sid+'.json');scores=read(f);assert sha(f.with_suffix('.npz'))==scores['matrices_sha256'];saved=np.load(f.with_suffix('.npz'));allmetrics[sid]={}
        for arm,info in r['arms'].items():
            assert sha(info['audio'])==info['audio_sha256']
            L=info['L'];a0=np.load(OUT/'audio_features'/sid/(arm+'.npz'))['audio'][:L]
            for cond in conds:
                i=np.arange(info['frames']);ix=i%132 if cond=='D0' else (i+66)%132 if cond=='D66' else np.full_like(i,0 if cond=='S0' else 66)
                assert np.array_equal(info['source_indices'][cond],ix)
                vrec=read(OUT/'features'/sid/arm/(cond+'.json'));assert sha(vrec['video']['path'])==vrec['video']['sha256'];assert np.array_equal(vrec['video']['source_indices'],ix);assert np.array_equal(vrec['video']['generation_boxes'],box[ix])
                for kind in ('generated','source_only'):
                    vf=OUT/'features'/sid/arm/(cond+'.npz') if kind=='generated' else OUT/'source_only'/(cond+'.npz');v0=np.load(vf)['visual'][:L]
                    for geom in ('raw','unit'):
                        v,a=v0,a0
                        if geom=='unit':v=(v.astype(float)/np.linalg.norm(v.astype(float),axis=1)[:,None]).astype(np.float32);a=(a.astype(float)/np.linalg.norm(a.astype(float),axis=1)[:,None]).astype(np.float32)
                        key=f'{kind}/{geom}/{arm}/{cond}';m=saved[key];ap=np.pad(a,((15,15),(0,0)))
                        direct=np.asarray([np.sqrt(((v.astype(float)-ap[15+lag:15+lag+L].astype(float)+1e-6)**2).sum(axis=1)) for lag in range(-15,16)]).T
                        errors['direct_distances']=max(errors['direct_distances'],float(abs(direct-m).max()));direct_count+=direct.size
                        allmetrics[sid][key]={pol:calculate(m,pol,read(OUT/'calibration.json')['k0']) for pol in ('guard20','valid','guard0')}
                        for pol in ('guard20','valid','guard0'):
                            z=calculate(m,pol,scores['k_at_scoring']);target=scores['cells'][key][pol]
                            for field in z:errors['score_metrics']=max(errors['score_metrics'],abs(z[field]-target[field]))
                        if r['split']=='calibration' and geom=='raw' and kind=='generated':cal_lags.append(allmetrics[sid][key]['guard20']['best_lag'])
        print('independent',sid,flush=True)
    assert int(np.median(cal_lags))==read(OUT/'calibration.json')['k0']
    checked=0
    for label,summary in read(OUT/'summary.json').items():
        subset,kind,geom,policy=label.split('/');selected=[r for r in rows if r['eligible'] and (subset=='all' or r['split']==subset)]
        values=defaultdict(lambda:defaultdict(list))
        for r in selected:
            for field in ('C','B','D','D_anchor','C_anchor'):
                gaps={c:allmetrics[r['id']][f'{kind}/{geom}/T/{c}'][policy][field]-allmetrics[r['id']][f'{kind}/{geom}/N/{c}'][policy][field] for c in conds}
                gaps.update(dynamic_minus_static=(gaps['D0']+gaps['D66']-gaps['S0']-gaps['S66'])/2,D66_minus_D0=gaps['D66']-gaps['D0'],S66_minus_S0=gaps['S66']-gaps['S0'])
                for name,value in gaps.items():values[f'{name}/{field}'][r['speaker']].append(value)
        for name,groups in values.items():
            y=np.array([np.mean(groups[s]) for s in sorted(groups)]);draws=np.random.Generator(np.random.PCG64(20260926)).integers(len(y),size=(20000,len(y)));ci=np.quantile(y[draws].mean(1),[.005,.995]);target=summary[name]
            errors['speaker_mean']=max(errors['speaker_mean'],abs(y.mean()-target['mean']));errors['ci99']=max(errors['ci99'],float(abs(ci-target['ci99']).max()));checked+=1
    assert errors['direct_distances']<2e-5 and errors['score_metrics']<1e-12 and errors['speaker_mean']<1e-12 and errors['ci99']<1e-12,errors
    assert read(OUT/'repeat_controls.json')['passed']
    assert read(OUT/'delay_controls.json')['passed']
    receipt={'passed':True,'errors':errors,'direct_distance_count':direct_count,'summaries_checked':checked,'cells_scored':len(rows)*32,'validator_sha256':sha(__file__),'score_lock_sha256':sha(OUT/'score_lock.json'),'note':'full vector distance, source mapping, frozen hashes, metric and speaker bootstrap independent reconstruction; no producer import'}
    (OUT/'independent_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
