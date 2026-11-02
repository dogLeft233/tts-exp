"""Independent NumPy reconstruction of native policy and probability sample."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import save,read,sha

def softmax(x):
    x=np.asarray(x,dtype=np.float64);v=np.exp(x-np.max(x,axis=-1,keepdims=True));return v/v.sum(axis=-1,keepdims=True)

def effective(raw,history,cb,temp=.9):
    x=raw.astype(np.float32).copy()
    if cb==0:
        indices=np.unique(np.asarray(history,dtype=int));v=x[indices]
        x[indices]=np.where(v<0,v*np.float32(1.05),v/np.float32(1.05))
        x[2048:]=-np.inf;x[2150]=raw[2150]
        if len(history)<2:x[2150]=-np.inf
    x/=np.float32(temp)
    threshold=np.partition(x,-50)[-50];x[x<threshold]=-np.inf
    return x

def main(run):
    report=[]
    for model in ['L06','L17']:
        d=run/('A_'+model);z=np.load(d/'native.npz');meta=read(d/'native.json');maxlog=maxprob=0.;sample={};samplemeta=[]
        for r in meta['rows']:
            cb=r['cb'];frame=r['frame'];idx=frame
            raw=z[f'raw_{cb}'][idx];eff=z[f'effective_{cb}'][idx];prob=z[f'probability_{cb}'][idx]
            got=effective(raw,r['history'],cb);assert np.array_equal(np.isfinite(got),np.isfinite(eff))
            mask=np.isfinite(eff);maxlog=max(maxlog,float(abs(got[mask]-eff[mask]).max()))
            p=softmax(got);maxprob=max(maxprob,float(abs(p-prob).max()))
            if frame<2 or cb==0 and frame==len(z['frame_0'])-1:
                key=f'f{frame}_cb{cb}'
                sample[key+'_raw']=raw;sample[key+'_effective']=eff;sample[key+'_probability']=prob;sample[key+'_mask']=np.isfinite(eff)
                nonzero=p>0;h=float(-(p[nonzero]*np.log(p[nonzero])).sum())
                samplemeta.append({**r,'key':key,'cpu_total_entropy':h})
        np.savez_compressed(run/(model+'_cpu_sample.npz'),**sample)
        save(run/(model+'_cpu_sample.json'),{'source_trace_sha':sha(d/'native.npz'),'rows':samplemeta,'EOS':2150,'content_IDs':[0,2047],'coarse_suppressed':'2048..3071 excluding2150','native_policy':'repetition1.05(coarse), min2(EOS), temp0.9, topk50, topp1.0'})
        assert maxlog<=1e-4 and maxprob<=1e-5
        report.append({'model':model,'rows':len(meta['rows']),'max_abs_policy_logits_CPU_rebuild':maxlog,'max_abs_probability_CPU_softmax':maxprob,'all_support_masks_exact':True,'status':'PASS'})
    save(run/'A_CPU_independent_audit.json',{'status':'PASS','rows':report,'implementation':'NumPy independently rebuilds repetition/special-mask/min-EOS/temperature/top-k and float64 softmax; no transformers or torch'})

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
