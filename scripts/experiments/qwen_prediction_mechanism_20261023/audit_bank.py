"""Independent CPU aggregation/bootstrap plus shared-history logit sample."""
from pathlib import Path
import argparse,json,hashlib,math,statistics
import numpy as np

def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def prob(x):
    x=np.asarray(x,dtype='float64');e=np.exp(x-np.max(x));return e/e.sum()
def h(p):return -math.fsum(float(v)*math.log(float(v)) for v in p if v>0)

def main(run):
    p=load(run/'B_protocol.json');g=load(run/'B_gate_and_calibration.json');diag=load(run/'B_history_diagnostics.json')['statistics'];stats={(r['reader'],r['id']):r for r in diag if 'reader' in r};speakers=sorted({r['sample_id'] for r in p['rows']});source={};checks=[]
    for model in ['L06','L17']:
        source[model]=[]
        for sid in speakers:
            books=[]
            for cb in range(16):
                origins=[]
                for origin in ['N','L06','L17']:
                    values=[stats[(model,r['id'])]['codebooks'][cb]['effective_content_entropy'] for r in p['rows'] if r['sample_id']==sid and r['source']==origin and r['qualified']]
                    origins.append(statistics.fmean(values))
                books.append(statistics.fmean(origins))
            source[model].append(books)
        delta=float(np.max(np.abs(np.asarray(source[model])-np.asarray(g['source_codebook_means'][model]))));assert delta<1e-12
        checks.append({'model':model,'max_abs_source_codebook_average_difference':delta})
    rng=np.random.default_rng(2026102307);indices=rng.integers(0,10,size=(20000,10))
    for name,cbs in [('coarse',[0]),('fine',list(range(1,16)))]:
        dif=[statistics.fmean(source['L17'][i][cb]-source['L06'][i][cb] for cb in cbs) for i in range(10)]
        mean=statistics.fmean(dif);boots=[statistics.fmean(dif[int(i)] for i in sample) for sample in indices];ci=np.quantile(boots,[.0125,.9875]);same=sum(v*(1 if mean>=0 else -1)>0 for v in dif)
        result=abs(mean)>=.10 and same>=8 and ci[0]*ci[1]>0
        assert abs(mean-g['gates'][name]['mean_L17_minus_L06_nat'])<1e-12 and np.max(abs(ci-np.array(g['gates'][name]['CI97_5'])))<1e-12 and bool(result)==g['gates'][name]['pass']
        checks.append({'category':name,'mean':mean,'CI97_5':ci.tolist(),'same_direction_sources':same,'gate_pass':bool(result)})
    for model in ['L06','L17']:
        own=next(r for r in p['rows'] if r['sample_id']==speakers[0] and r['source']==model and r['draw']==1)
        a=np.load(run/('A_'+model)/'native.npz');b=np.load(run/('B_'+model)/own['id']/'trace.npz');max_raw=max_prob=0
        for cb in range(16):
            max_raw=max(max_raw,float(np.max(abs(a[f'raw_{cb}']-b[f'raw_{cb}']))))
            max_prob=max(max_prob,float(np.max(abs(a[f'probability_{cb}']-b[f'probability_{cb}']))))
        assert max_raw<=1e-4 and max_prob<=1e-5
        checks.append({'model':model,'A_native_vs_B_own_complete_history_raw_max_abs':max_raw,'probability_max_abs':max_prob,'decoder_bypass_does_not_change_predictions':True})
    row=p['rows'][0];targets=np.load(row['codes']['path']);meta=[];probabilities={};max_delta=0
    for model in ['L06','L17']:
        directory=run/('B_'+model)/row['id'];z=np.load(directory/'trace.npz');capture=load(directory/'trace.json');arrays={};probabilities[model]={}
        for frame in [0,1]:
            for cb in range(16):
                key=f'f{frame}_cb{cb}';raw=z[f'raw_{cb}'][frame];eff=z[f'effective_{cb}'][frame];pt=z[f'probability_{cb}'][frame];pcpu=prob(eff)
                max_delta=max(max_delta,float(max(abs(pcpu-pt))));cp=prob(eff[:2048]);rp=prob(raw[:2048]);target=int(targets[frame,cb]);true=float(cp[target]);rawtrue=float(rp[target])
                arrays[key+'_raw']=raw;arrays[key+'_effective']=eff;arrays[key+'_probability']=pt;arrays[key+'_mask']=np.isfinite(eff)
                history=next(r['history'] for r in capture['rows'] if r['frame']==frame and r['cb']==cb)
                meta.append({'model':model,'frame':frame,'cb':cb,'key':key,'actual_target_code':target,'input_ID_history':history,'effective_content_entropy':h(cp),'raw_content_entropy':h(rp),'raw_NLL':-math.log(rawtrue),'effective_NLL':-math.log(true) if true>0 else '+inf','raw_target_rank':int(1+sum(raw[:2048]>raw[target]))})
                probabilities[model][key]=(rp,cp)
        np.savez_compressed(run/('B_CPU_common_sample_'+model+'.npz'),**arrays)
    js=[]
    for key in probabilities['L06']:
        r={'key':key}
        for i,name in enumerate(['raw','effective']):
            a,b=probabilities['L06'][key][i],probabilities['L17'][key][i];r[name+'_JSD']=h((a+b)/2)-(h(a)+h(b))/2;r[name+'_top1_disagree']=bool(a.argmax()!=b.argmax())
        js.append(r)
    assert max_delta<1e-5
    write(run/'B_CPU_common_sample.json',{'history_id':row['id'],'history_codes_sha':row['codes']['sha256'],'complete_forced_codes':targets.tolist(),'rows':meta,'shared_history_JSD':js,'max_CPU_softmax_difference':max_delta,'same_complete_history_for_both_models':True})
    write(run/'B_independent_CPU_audit.json',{'status':'PASS','aggregation_and_bootstrap':checks,'bootstrap':'independent Python statistics.fmean aggregation with same frozen NumPy random indices','common_history_probability_max_abs_difference':max_delta,'sample_records':len(meta),'sample_frames':2,'independent_unit_sources':10})

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
