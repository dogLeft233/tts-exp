"""Independent scalar-distance/candidate/bootstrap audit, not using oracle helpers."""
import math
from pathlib import Path
import numpy as np
from scripts.experiments.tts_independent_visual import ROOT,read,write,sha

def main():
    out=ROOT/'runs/tts_instance_local_oracle_20260926';p=read(out/'protocol.json');src=Path(p['source'])
    for f,h in p['assets'].items():assert sha(f)==h
    maxerr=0.;baseline=0.;count=0
    for label in ('primary','mfa_only'):
        for k in (3,2,4):
            folder=out/f'{label}_k{k}';ss={s['id']:s for s in read(src/'mechanism'/f'{label}_k{k}'/'support.json') if s['eligible']}
            old={(c['id'],c['geometry']):c for c in read(src/'mechanism'/f'{label}_k{k}'/'cells.json') if c['image']=='3'}
            cache={};records=read(folder/'queries.json');bycell={}
            for r in records:
                sid=r['id'];geo=r['geometry'];s=ss[sid];node=s['nodes'][s['queries'][r['query']]['node']]
                if (sid,geo) not in cache:
                    z=np.load(src/'audio_features'/(sid+'.npz'));aa={a:z[a].astype(float) for a in ('Q1','Q2')};vv={a:np.load(src/'scores/3'/sid/(a+'.npz'))['visual'].astype(float) for a in aa}
                    if geo=='unit':
                        aa={a:x/np.sqrt(np.sum(x*x,axis=1))[:,None] for a,x in aa.items()};vv={a:x/np.sqrt(np.sum(x*x,axis=1))[:,None] for a,x in vv.items()}
                    cache[sid,geo]=(aa,vv)
                aa,vv=cache[sid,geo];eff={}
                for a,b in (('Q1','Q2'),('Q2','Q1')):
                    d=r['directions'][a+b];j0=node['j'][b];lo,hi=node['span'][b];radius,same=p['modes'][r['mode']]
                    js=[j for j in range(s['lengths'][b]) if 20<=j-k<s['lengths'][b]-20 and (radius is None or abs(j-j0)<=radius) and (not same or lo<=j/25+.1075<hi)]
                    assert js==d['candidates'] and j0 in js
                    v=vv[a][node['j'][a]-k]
                    ds=[math.sqrt(math.fsum(float(x-y+1e-6)**2 for x,y in zip(v,aa[b][j]))) for j in js]
                    pos=math.sqrt(math.fsum(float(x-y+1e-6)**2 for x,y in zip(v,aa[a][node['j'][a]])))
                    ix=min(range(len(js)),key=lambda i:(ds[i],abs(js[i]-j0),js[i]));assert js[ix]==d['chosen']
                    maxerr=max(maxerr,abs(ds[ix]-d['cross']),abs(pos-d['self']));count+=len(js)
                    eff[a+b+'_gap']=ds[ix]-pos
                eff['gap']=(eff['Q1Q2_gap']+eff['Q2Q1_gap'])/2
                for name,x in eff.items():maxerr=max(maxerr,abs(x-r['effects'][name]))
                key=(sid,geo,r['mode']);bycell.setdefault(key,[]).append(eff)
            cells=read(folder/'cells.json')
            for c in cells:
                vals=bycell[c['id'],c['geometry'],c['mode']];assert len(vals)==c['queries']==len(ss[c['id']]['queries'])
                for name in ('gap','Q1Q2_gap','Q2Q1_gap'):maxerr=max(maxerr,abs(np.mean([v[name] for v in vals])-c['effects'][name]))
                if c['mode']=='r0':baseline=max(baseline,abs(c['effects']['gap']+old[c['id'],c['geometry']]['effects']['positive_STT']))
            for geo,ms in read(folder/'analysis.json').items():
                for mode,z in ms.items():
                    cc=[c for c in cells if c['geometry']==geo and c['mode']==mode];sp=sorted({c['speaker'] for c in cc})
                    for name in ('gap','Q1Q2_gap','Q2Q1_gap'):
                        means=np.array([np.mean([c['effects'][name] for c in cc if c['speaker']==s]) for s in sp]);rng=np.random.Generator(np.random.PCG64(20260926));boot=means[rng.integers(len(sp),size=(20000,len(sp)))].mean(1)
                        maxerr=max(maxerr,abs(means.mean()-z[name]['speaker_mean']),float(np.max(abs(np.quantile(boot,[.005,.995])-z[name]['speaker_ci99']))))
    result={'candidate_distances_checked':count,'independent_scalar_max_error':maxerr,'r0_original_positive_replay_error':baseline,'PASS':maxerr<1e-10 and baseline<1e-10,'source_assets_verified':len(p['assets'])}
    write(out/'validation.json',result);print(result);assert result['PASS']

if __name__=='__main__':main()
