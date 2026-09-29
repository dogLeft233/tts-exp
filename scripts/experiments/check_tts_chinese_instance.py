"""Independent arithmetic replay from frozen arrays; no scoring helper imports."""
import json
from pathlib import Path
import numpy as np
from scripts.experiments.tts_chinese_instance import BASE,read,write,sha

def main():
    worst_matrix=0.;worst_event=0.;n_matrix=0;n_event=0;confidences={}
    p=read(BASE/'protocol.json');source={r['id']:r for r in p['rows']}
    calibration_lags=[]
    for f in (BASE/'scores').glob('*/*/*.npz'):
        im,sid,a=f.parts[-3],f.parts[-2],f.stem
        z=np.load(f);v=z['visual'].astype(float);m=z['matrix'];n=len(m)
        aud=np.load(BASE/'audio_features'/(sid+'.npz'))[a][:n].astype(float);pad=np.pad(aud,((15,15),(0,0)))
        rebuilt=np.column_stack([np.sqrt(((v[:n]-pad[j:j+n]+1e-6)**2).sum(-1)) for j in range(31)])
        worst_matrix=max(worst_matrix,float(abs(rebuilt-m).max()));n_matrix+=1
        rec=read(f.with_suffix('.json'))
        for g,metrics in rec['metrics'].items():
            guard=int(g);curve=rebuilt[guard:n-guard or None].mean(0);conf=np.median(curve)-min(curve)
            assert abs(conf-metrics['C'])<1e-5
            confidences[(im,sid,g,a)]=float(conf)
            if im=='3' and source[sid]['split']=='calibration' and a in ('N','Q1','Q2') and guard==20:calibration_lags.append(int(curve.argmin())-15)
    assert int(np.median(calibration_lags))==read(BASE/'calibration.json')['k']
    native=read(BASE/'native.json');native_max=0.
    for im,obj in native.items():
        independently=[]
        for r in p['rows']:
            if r['split']!='evaluation':continue
            effects={}
            for g in ('0','15','20'):
                if not all((im,r['id'],g,a) in confidences for a in ('N','C','Q1','Q2')):continue
                cs={a:confidences[(im,r['id'],g,a)] for a in ('N','C','Q1','Q2')}
                gq=(cs['Q1']+cs['Q2'])/2-cs['N'];gc=cs['C']-cs['N']
                effects.update({g+'_GQ':gq,g+'_GC':gc,g+'_GQ-GC':gq-gc,**{g+'_level_'+a:v for a,v in cs.items()}})
            independently.append({'speaker':r['speaker'],'effects':effects})
        for name,z in obj['analysis'].items():
            subset=[r for r in independently if name in r['effects']];speakers=sorted({r['speaker'] for r in subset})
            means=np.array([np.mean([r['effects'][name] for r in subset if r['speaker']==sp]) for sp in speakers]);rng=np.random.Generator(np.random.PCG64(20260926));bs=means[rng.integers(len(means),size=(20000,len(means)))].mean(1)
            native_max=max(native_max,abs(means.mean()-z['speaker_mean']),float(np.max(abs(np.quantile(bs,[.005,.995])-z['speaker_ci99']))));assert len(subset)==z['n']
    for folder in (BASE/'mechanism').iterdir():
        if not folder.is_dir():continue
        k=int(folder.name.rsplit('k',1)[1]);supports={s['id']:s for s in read(folder/'support.json')}
        for cell in read(folder/'cells.json'):
            sid=cell['id'];s=supports[sid];arms=list(s['lengths']);im=cell['image'];geo=cell['geometry']
            audio=np.load(BASE/'audio_features'/(sid+'.npz'));aa={a:audio[a].astype(float) for a in arms}
            vv={a:np.load(BASE/'scores'/im/sid/(a+'.npz'))['visual'].astype(float) for a in arms}
            if geo=='unit':
                aa={a:x/np.sqrt((x*x).sum(-1))[:,None] for a,x in aa.items()};vv={a:x/np.sqrt((x*x).sum(-1))[:,None] for a,x in vv.items()}
            calc={}
            for v in arms:
                for a in arms:
                    vals=[]
                    for query in s['queries']:
                        q=s['nodes'][query['node']];i=q['j'][v]-k;j=q['j'][a]
                        assert 20<=i<s['lengths'][v]-20 and len(query['donors'])>=5
                        donor=[s['nodes'][d]['j'][a] for d in query['donors']];assert all(3<=abs(j-d)<=15 for d in donor)
                        positive=np.linalg.norm(vv[v][i]-aa[a][j]+1e-6)
                        neg=np.linalg.norm(vv[v][i][None]-aa[a][donor]+1e-6,axis=1)
                        vals.append((positive,neg.mean(),neg.mean()-positive,((neg>positive)+.5*(neg==positive)).mean()));n_event+=1
                    calc[v+a]=np.asarray(vals).mean(0)
            for ix,metric in enumerate(('positive','negative','margin','rank')):
                for pair,val in calc.items():worst_event=max(worst_event,abs(float(val[ix])-cell['effects'][metric+'_q_'+pair]))
                t=.5*(calc['Q1Q1'][ix]+calc['Q2Q2'][ix]-calc['Q1Q2'][ix]-calc['Q2Q1'][ix])
                n=.25*(2*calc['NN'][ix]+calc['Q1Q1'][ix]+calc['Q2Q2'][ix]-calc['NQ1'][ix]-calc['Q1N'][ix]-calc['NQ2'][ix]-calc['Q2N'][ix])
                diag=(calc['Q1Q1'][ix]+calc['Q2Q2'][ix])/2-calc['NN'][ix]
                off=(calc['Q1Q2'][ix]+calc['Q2Q1'][ix])/2-calc['NN'][ix]
                assert abs(diag-off-t)<1e-10
                for name,val in [('STT',t),('SNT',n),('SNT-STT',n-t),('Tdiag-NN',diag),('Toffdiag-NN',off)]:worst_event=max(worst_event,abs(float(val)-cell['effects'][metric+'_'+name]))
        analysis=read(folder/'analysis.json');cells=read(folder/'cells.json')
        for im,gs in analysis.items():
            for geo,results in gs.items():
                subset=[c for c in cells if c['image']==im and c['geometry']==geo]
                for name,res in results.items():
                    speakers=sorted({r['speaker'] for r in subset});means=np.asarray([np.mean([r['effects'][name] for r in subset if r['speaker']==sp]) for sp in speakers])
                    rng=np.random.Generator(np.random.PCG64(20260926));samples=means[rng.integers(len(means),size=(20000,len(means)))].mean(1)
                    assert abs(means.mean()-res['speaker_mean'])<1e-10
                    assert np.max(abs(np.quantile(samples,[.005,.995])-res['speaker_ci99']))<1e-10
    result={'matrix_count':n_matrix,'event_query_pair_count':n_event,'matrix_max_error':worst_matrix,'event_max_error':worst_event,'native_mean_ci_max_error':native_max,'bootstrap':'independently rebuilt speaker weights/PCG64 99CI for native and event','PASS':worst_matrix<1e-5 and worst_event<1e-10 and native_max<1e-5}
    write(BASE/'validation.json',result);print(result);assert result['PASS']

if __name__=='__main__':main()
