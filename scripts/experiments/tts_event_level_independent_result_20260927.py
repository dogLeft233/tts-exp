"""Read-only independent cache-event recomputation; no production imports."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/tts_level_event_cross_20260927'
OUT=ROOT/'runs/tts_event_level_independent_audit_20260927'
FIX=ROOT/'runs/tts_fixed_level_generation_cross_20260927/features'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    protocol=read(RUN/'protocol.json'); idx=read(RUN/'indices.json')
    assert read(RUN/'execution.json')['status']=='SCORED_PENDING_INDEPENDENT'
    assert read(RUN/'review.json')['protocol_sha256']==sha(RUN/'protocol.json')
    for path,digest in {**read(RUN/'inputs.json'),**protocol['code']}.items():
        assert sha(path)==digest,path
    for name,digest in read(RUN/'seal.json').items(): assert sha(RUN/name)==digest,name
    bridge=read(RUN/'bridge_validation.json');assert bridge['status']=='PASS'
    assert bridge['protocol_sha256']==sha(RUN/'protocol.json')
    for k in ['old_query_max','old_statistic_max','new_raw_image3_query_max']:assert bridge[k]<=1e-9
    assert (RUN/'review.json').stat().st_mtime < (RUN/'bridge_validation.json').stat().st_mtime < (RUN/'query_metrics.npz').stat().st_mtime
    with np.load(RUN/'query_metrics.npz') as f: expected=f['metrics']
    calc=np.zeros_like(expected)
    assert calc.shape==(1411,2,20,4)
    for row,span in zip(idx['rows'],idx['spans']):
        audio={};video={}
        for state in ('raw','FIXED','LEVEL'):
            for arm in 'NT':
                with np.load(FIX/row['id']/arm/(state+'.npz')) as f:
                    audio[state,arm]=f['audio'];video[state,arm]=f['visual']
        for gi in range(2):
            # Historical float32 normalization is part of the estimand.
            A={k:(x/np.linalg.norm(x,axis=1)[:,None] if gi else x).astype('float64') for k,x in audio.items()}
            V={k:(x/np.linalg.norm(x,axis=1)[:,None] if gi else x).astype('float64') for k,x in video.items()}
            for ci,c in enumerate(protocol['cells']):
                av=A[c['audio_state'],c['audio_source']];vv=V[c['video_state'],c['video_source']]
                for qi,q in enumerate(row['queries']):
                    node=row['nodes'][q['node']]
                    donor_nodes=[q['node']]+q['donors']
                    aix=np.array([row['nodes'][n]['j'][c['audio_source']] for n in donor_nodes])
                    vx=vv[node['j'][c['video_source']]-3]
                    ds=np.linalg.norm((vx[None,:]-av[aix])+1e-6,axis=1)
                    pos=ds[0]; neg=ds[1:]; mean=neg.mean()
                    rank=(np.count_nonzero(neg>pos)+np.count_nonzero(neg==pos)/2)/len(neg)
                    calc[span['start']+qi,gi,ci]=[pos,mean,mean-pos,rank]
    qe=float(np.max(abs(calc-expected)));assert qe<1e-12,qe
    cm=np.array([calc[s['start']:s['stop']].mean(axis=0) for s in idx['spans']])
    with np.load(RUN/'clip_metrics.npz') as f: ce=float(np.max(abs(cm-f['metrics'])))
    assert ce<1e-12
    speakers=[r['speaker'] for r in idx['spans']]; names=sorted(set(speakers))
    counts=np.array([speakers.count(s) for s in names]); membership=np.array([[a==s for a in speakers] for s in names])
    draws=np.random.default_rng(20260926).integers(13,size=(20000,13))
    source_weights={'visual_at_N_audio':[-1,0,1,0],'audio_at_N_visual':[-1,1,0,0],
       'interaction':[1,-1,-1,1],'visual_at_T_audio':[0,-1,0,1],
       'audio_at_T_visual':[0,0,-1,1],'diagonal':[-1,0,0,1]}
    state_weights={'G':[-1,1,0,0],'E':[-1,0,1,0],'I':[1,-1,-1,1],'total':[-1,0,0,1]}
    summary=read(RUN/'summary.json'); effects=np.load(RUN/'clip_effects.npz')
    maximum=0.;effect_error=0.;keys=[];n_scalar=0
    def compare(key,values):
        nonlocal maximum,effect_error,n_scalar
        effect_error=max(effect_error,float(np.max(abs(values-effects[key]))))
        means=np.array([values[mask].mean() for mask in membership]);sim=means[draws]
        weighted=(sim*counts[draws]).sum(axis=1)/counts[draws].sum(axis=1)
        got={'n':32,'speakers':13,'speaker_mean':means.mean(),'speaker_ci95':np.quantile(sim.mean(1),[.025,.975]),
            'speaker_ci99':np.quantile(sim.mean(1),[.005,.995]),'utterance_mean':values.mean(),
            'utterance_cluster_ci95':np.quantile(weighted,[.025,.975]),'positive':int((values>0).sum()),
            'speaker_positive':int((means>0).sum()),'per_speaker':dict(zip(names,means))}
        for field,value in got.items():
            if field=='per_speaker':
                errs=[abs(value[s]-summary[key][field][s]) for s in names]
            else:errs=np.abs(np.asarray(value)-np.asarray(summary[key][field])).ravel()
            maximum=max(maximum,float(np.max(errs)));n_scalar+=len(errs)
        keys.append(key)
    for gi,g in enumerate(('raw','unit')):
        for mi,m in enumerate(('positive','negative','margin','rank')):
            x=cm[:,gi,:,mi].reshape(32,5,4); series={}
            for h,hn in enumerate(('RR','FR','RF','FF','LL')):
                for s,sn in enumerate(('NN','NT','TN','TT')):series['cell/'+hn+'_'+sn]=x[:,h,s]
                for name,w in source_weights.items():series['source/'+hn+'/'+name]=sum(x[:,h,j]*a for j,a in enumerate(w))
            for s,sn in enumerate(('NN','NT','TN','TT')):
                for name,w in state_weights.items():series['state/'+sn+'/'+name]=sum(x[:,h,s]*a for h,a in enumerate(w))
                series['LEVEL_minus_RAW/'+sn]=x[:,4,s]-x[:,0,s]
            for name in state_weights:series['state_gap/'+name]=series['state/TT/'+name]-series['state/NN/'+name]
            series['LEVEL_minus_RAW/diagonal']=(x[:,4,3]-x[:,4,0])-(x[:,0,3]-x[:,0,0])
            for key,value in series.items():compare(g+'/'+m+'/'+key,value)
    assert set(keys)==set(summary)==set(effects.files) and len(keys)==600
    assert maximum<1e-11 and effect_error<1e-12,(maximum,effect_error)
    receipts={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(RUN/'protocol.json'),
      'auditor_code_sha256':sha(__file__),'producer_inputs_verified':len(read(RUN/'inputs.json')),
      'query_entries':int(calc.size),'query_max_error':qe,'clip_max_error':ce,'effect_max_error':effect_error,
      'summary_endpoints':len(keys),'summary_scalars':n_scalar,'statistics_max_error':maximum,
      'historical_bridge_status':'PASS','review_before_bridge_before_scores':True,
      'all_legacy_queries':1411,'all_clips':32,'all_speakers':13,'no_GPU_or_model_or_support_selection':True,
      'result_hashes':{n:sha(RUN/n) for n in ('query_metrics.npz','clip_metrics.npz','clip_effects.npz','summary.json','execution.json','bridge_validation.json')}}
    dest=OUT/'result_receipt.json';assert not dest.exists();dest.write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'status':'PASS','receipt':str(dest),'sha256':sha(dest),'max_error':maximum}))

if __name__=='__main__':main()
