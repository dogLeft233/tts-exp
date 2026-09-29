"""Independent DTW optimality, sealed index, scalar score and bootstrap replay."""
import bisect
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scripts.experiments.tts_instance_audio_dtw import ROOT,OUT,SOURCE,read,write,sha

def main():
    protocol=read(OUT/'protocol.json');audioseal=read(OUT/'audio_mapping_seal.json');qseal=read(OUT/'query_mapping_seal.json');start=read(OUT/'scoring_start.json')
    assert sha(OUT/'audio_mapping_seal.json')==qseal['audio_mapping_seal_sha256']==start['audio_mapping_seal_sha256']
    assert sha(OUT/'query_mapping_seal.json')==start['query_mapping_seal_sha256']
    assets={**protocol['audio_assets'],**audioseal['assets'],**qseal['support_assets'],**qseal['mapping_assets'],**read(OUT/'scoring_inputs.json')}
    for f,h in assets.items():assert sha(f)==h
    worstcost=0.;worstscore=0.;checkedpaths=0;indexcount=0
    for row in protocol['rows']:
        mapping=read(OUT/'maps'/(row['id']+'.json'));mel={}
        for a in ('Q1','Q2'):
            z=np.load(OUT/'mel'/row['id']/(a+'.npz'));x=z['raw'];mu=np.sum(x,axis=0)/len(x);sd=np.sqrt(np.sum((x-mu)**2,axis=0)/len(x));rebuilt=(x-mu)/np.maximum(sd,1e-8);assert np.max(abs(rebuilt-z['cmvn']))<1e-12;mel[a]=rebuilt
        for phone in mapping['phones']:
            path=np.asarray(phone['path'])
            if not len(path):continue
            lo1,hi1=phone['spans'][0];lo2,hi2=phone['spans'][1]
            ix=[i for i in range(len(mel['Q1'])) if lo1<=i/80<hi1];iy=[i for i in range(len(mel['Q2'])) if lo2<=i/80<hi2]
            assert list(path[0])==[ix[0],iy[0]] and list(path[-1])==[ix[-1],iy[-1]]
            assert all(tuple(s) in ((1,0),(0,1),(1,1)) for s in np.diff(path,axis=0))
            costs=cdist(mel['Q1'][ix],mel['Q2'][iy],metric='sqeuclidean');dp=np.full(costs.shape,np.inf)
            for i in range(len(ix)):
                for j in range(len(iy)):
                    prev=[dp[i-1,j]] if i else []
                    if j:prev.append(dp[i,j-1])
                    if i and j:prev.append(dp[i-1,j-1])
                    dp[i,j]=costs[i,j]+(min(prev) if prev else 0)
            summed=sum(costs[i-ix[0],j-iy[0]] for i,j in path)
            worstcost=max(worstcost,abs(dp[-1,-1]-phone['cost']),abs(summed-dp[-1,-1]));checkedpaths+=1
    for label in ('primary','mfa_only'):
        for k in (3,2,4):
            rows=read(OUT/'indices'/f'{label}_k{k}.json');supports={s['id']:s for s in read(SOURCE/'mechanism'/f'{label}_k{k}'/'support.json') if s['eligible']};maps={};embedding={};replayed={}
            scores={(r['id'],r['query'],r['geometry']):r for r in read(OUT/f'{label}_k{k}/queries.json')}
            for r in rows:
                sid=r['id'];s=supports[sid];node=s['nodes'][s['queries'][r['query']]['node']]
                if sid not in maps:maps[sid]={p['event']:p for p in read(OUT/'maps'/(sid+'.json'))['phones']}
                for d,z in r['directions'].items():
                    a,b=('Q1','Q2') if d=='Q1Q2' else ('Q2','Q1');src=0 if a=='Q1' else 1;path=maps[sid][node['event']]['path'];knots=sorted({p[src] for p in path});target=[sum(p[1-src] for p in path if p[src]==i)/sum(p[src]==i for p in path)/80 for i in knots];xx=[x/80 for x in knots];t=node['j'][a]/25+.1075
                    ind=bisect.bisect_right(xx,t)
                    expected=target[0] if ind==0 else target[-1] if ind==len(xx) else target[ind-1]+(target[ind]-target[ind-1])*(t-xx[ind-1])/(xx[ind]-xx[ind-1])
                    assert abs(expected-z['target_time'])<1e-12
                    j0=node['j'][b];lo,hi=node['span'][b];js=[j for j in range(s['lengths'][b]) if abs(j-j0)<=2 and 20<=j-k<s['lengths'][b]-20 and lo<=j/25+.1075<hi]
                    assert js==z['candidate_indices'] and z['chosen_j']==min(js,key=lambda j:(abs(j/25+.1075-expected),abs(j-j0),j));assert z['self_j']==node['j'][a] and z['visual_i']==node['j'][a]-k;indexcount+=1
                if sid not in embedding:
                    az=np.load(SOURCE/'audio_features'/(sid+'.npz'));aa={a:az[a].astype(float) for a in ('Q1','Q2')};vv={a:np.load(SOURCE/'scores/3'/sid/(a+'.npz'))['visual'].astype(float) for a in aa};embedding[sid]=(aa,vv)
                for geo in ('raw','unit'):
                    aa,vv=embedding[sid]
                    if geo=='unit':aa={a:x/np.sqrt(np.einsum('ij,ij->i',x,x))[:,None] for a,x in aa.items()};vv={a:x/np.sqrt(np.einsum('ij,ij->i',x,x))[:,None] for a,x in vv.items()}
                    eff={}
                    for d,z in r['directions'].items():
                        a,b=('Q1','Q2') if d=='Q1Q2' else ('Q2','Q1');v=vv[a][z['visual_i']];distances=[]
                        for arm,j in ((a,z['self_j']),(b,z['original_j']),(b,z['chosen_j'])):
                            delta=v-aa[arm][j]+1e-6;distances.append(float(np.sqrt(np.dot(delta,delta))))
                        selfd,mfa,dt=distances;eff.update({d+'_dtw_gap':dt-selfd,d+'_mfa_gap':mfa-selfd,d+'_dtw_minus_mfa':dt-mfa})
                    for m in ('dtw_gap','mfa_gap','dtw_minus_mfa'):eff[m]=(eff['Q1Q2_'+m]+eff['Q2Q1_'+m])/2
                    for m,x in eff.items():worstscore=max(worstscore,abs(x-scores[sid,r['query'],geo]['effects'][m]))
                    replayed.setdefault((sid,geo),[]).append(eff)
            cells=read(OUT/f'{label}_k{k}/cells.json');analysis=read(OUT/f'{label}_k{k}/analysis.json')
            for c in cells:
                rr=replayed[c['id'],c['geometry']];assert len(rr)==c['queries']==len(supports[c['id']]['queries'])
                for m in rr[0]:worstscore=max(worstscore,abs(np.mean([r[m] for r in rr])-c['effects'][m]))
            for geo in ('raw','unit'):
                cc=[c for c in cells if c['geometry']==geo];sp=sorted({c['speaker'] for c in cc})
                for m in ('dtw_gap','mfa_gap','dtw_minus_mfa'):
                    means=np.array([np.mean([c['effects'][m] for c in cc if c['speaker']==s]) for s in sp]);rng=np.random.Generator(np.random.PCG64(20260926));bs=means[rng.integers(len(sp),size=(20000,len(sp)))].mean(1);z=analysis[geo][m]
                    worstscore=max(worstscore,abs(means.mean()-z['speaker_mean']),float(np.max(abs(np.quantile(bs,[.005,.995])-z['speaker_ci99']))))
    control=read(OUT/'controls.json');identitymax=max(r['identity_error_ms'] for r in control['records']);assert identitymax==0
    for phone in control['identity_paths']['phones']:
        assert all(a==b for a,b in phone['path']) and phone['cost']==0
    for r in control['records']:
        span=control['warp_paths']['phones'][r['event']]['spans'][0];lo,hi=[round(v*16000)/16000 for v in span];sx=[lo,(lo+hi)/2,hi];sy=[lo,lo+.75*(hi-lo),hi]
        truth=float(np.interp(r['source_time'],sx,sy)) if r['direction']=='Q1Q2' else float(np.interp(r['source_time'],sy,sx));assert abs(truth-r['truth_time'])<1e-12
    result={'assets_verified':len(assets),'phone_paths_optimality_checked':checkedpaths,'path_cost_max_error':worstcost,'mapped_query_directions_checked':indexcount,'score_and_bootstrap_max_error':worstscore,'identity_max_error_ms':identitymax,'known_warp_truth_independently_checked':True,'PASS':worstcost<1e-6 and worstscore<1e-10}
    write(OUT/'validation.json',result);print(result);assert result['PASS']

if __name__=='__main__':main()
