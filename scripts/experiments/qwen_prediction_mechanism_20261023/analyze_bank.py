"""Prespecified source-balanced diagnostics and entropy-only temperatures."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha
from scripts.experiments.qwen_prediction_mechanism_20261023.cpu_audit import softmax

def entropy(p):
    return -np.sum(np.where(p>0,p*np.log(np.maximum(p,np.finfo(float).tiny)),0),axis=-1)

def average(rows,values,qualified=True):
    """Returns ten source means, equal N/T06/T17, then equal draws."""
    answer=[]
    for sid in sorted({r['sample_id'] for r in rows}):
        groups=[]
        for src in ['N','L06','L17']:
            group=[values[r['id']] for r in rows if r['sample_id']==sid and r['source']==src and (r['qualified'] or not qualified)]
            assert group,(sid,src)
            groups.append(np.mean(group,axis=0))
        answer.append(np.mean(groups,axis=0))
    return np.array(answer)

def interval(x,seed=2026102307):
    x=np.asarray(x);rng=np.random.default_rng(seed);boots=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(boots,[.0125,.9875])]

def match_temperature(value,target):
    lo,hi=.4,1.6;vl,vh=value(lo),value(hi)
    base={'target':target,'endpoint_entropy':[vl,vh],'bounds':[lo,hi]}
    if vl>vh:return {**base,'status':'UNMATCHABLE','reason':'nonmonotone endpoint entropy'}
    if target<vl or target>vh:
        temp,achieved=(lo,vl) if target<vl else (hi,vh)
        return {**base,'status':'MATCHED' if abs(achieved-target)<=.02 else 'UNMATCHABLE','temperature':temp,'achieved':achieved,'error':achieved-target,'solution':'nearest bounded endpoint; same frozen0.02nat error criterion'}
    for _ in range(30):
        mid=(lo+hi)/2;vm=value(mid)
        if vm<target:lo=mid
        else:hi=mid
    temp=(lo+hi)/2;achieved=value(temp)
    return {**base,'status':'MATCHED' if abs(achieved-target)<=.02 else 'UNMATCHABLE','temperature':temp,'achieved':achieved,'error':achieved-target,'solution':'bounded deterministic bisection'}

def main(run):
    p=read(run/'B_protocol.json');rows=p['rows'];stats=[];effs={'L06':{},'L17':{}};metrics={'L06':{},'L17':{}}
    for model in ['L06','L17']:assert read(run/f'B_{model}/completion.json')['count']==70
    for row in rows:
        records={};target=np.load(row['codes']['path']);T=len(target)
        for model in ['L06','L17']:
            z=np.load(run/f'B_{model}'/row['id']/'trace.npz');effs[model][row['id']]=[];records[model]=[]
            codebook=[]
            for cb in range(16):
                raw=z[f'raw_{cb}'][:T,:2048].astype(float);eff=z[f'effective_{cb}'][:T,:2048].astype(float)
                assert raw.shape==(T,2048) and eff.shape==(T,2048)
                assert np.isfinite(raw).all() and not np.isnan(eff).any() and np.isfinite(eff).any(axis=1).all()
                pr=softmax(raw);pe=softmax(eff);effs[model][row['id']].append(eff)
                assert np.isfinite(pr).all() and np.isfinite(pe).all()
                ids=target[:,cb];ix=np.arange(T);actual=pe[ix,ids];rawactual=pr[ix,ids]
                m={'cb':cb,'raw_entropy':float(entropy(pr).mean()),'effective_content_entropy':float(entropy(pe).mean()),'raw_NLL':float(-np.log(rawactual).mean()),'effective_NLL':float(-np.log(actual).mean()) if np.all(actual>0) else '+inf','zero_policy_probability_fraction':float(np.mean(actual==0)),'raw_target_rank':float(np.mean(1+(raw>raw[ix,ids,None]).sum(axis=1)))}
                codebook.append(m);records[model].append((pr,pe))
            coarseprob=z['probability_0'];eos=coarseprob[:,2150]
            stat={'id':row['id'],'sample_id':row['sample_id'],'source':row['source'],'draw':row['draw'],'reader':model,'qualified':row['qualified'],'frames':T,'codebooks':codebook,'EOS_probability_content_frames_mean':float(eos[:T].mean()),'EOS_probability_terminal':float(eos[T]),'terminal_observed':row['terminal_EOS_observed'],'total_effective_coarse_entropy_content_frames':float(entropy(coarseprob[:T]).mean())}
            stats.append(stat);metrics[model][row['id']]=[m['effective_content_entropy'] for m in codebook]
        comparison=[]
        for cb in range(16):
            a,b=records['L06'][cb],records['L17'][cb];item={'cb':cb}
            for key,i in [('raw',0),('effective',1)]:
                x,y=a[i],b[i];mid=(x+y)/2
                js=entropy(mid)-(entropy(x)+entropy(y))/2
                item[key+'_JSD']=float(js.mean());item[key+'_top1_disagreement']=float(np.mean(x.argmax(1)!=y.argmax(1)))
            comparison.append(item)
        stats.append({'id':row['id'],'comparison':comparison})
    source={m:average(rows,metrics[m]) for m in metrics};difference=source['L17']-source['L06'];gates={}
    for name,cols in [('coarse',[0]),('fine',list(range(1,16)))]:
        dif=difference[:,cols].mean(1);mean=float(dif.mean());ci=interval(dif);same=int(np.sum(dif*np.sign(mean)>0));passed=abs(mean)>=.10 and same>=8 and ci[0]*ci[1]>0
        gates[name]={'mean_L17_minus_L06_nat':mean,'CI97_5':ci,'same_direction_sources':same,'source_differences':dif.tolist(),'pass':bool(passed)}
    calibrated={}
    for model,other in [('L06','L17'),('L17','L06')]:
        calibrated[model]={}
        for name,cols in [('coarse',[0]),('fine',list(range(1,16)))]:
            target=float(source[other][:,cols].mean())
            def value(temp):
                values={r['id']:[float(entropy(softmax(effs[model][r['id']][cb]*(.9/temp))).mean()) for cb in cols] for r in rows if r['qualified']}
                return float(average(rows,values).mean())
            if not gates[name]['pass']:
                calibrated[model][name]={'status':'NOT_ADVANCED','temperature':.9,'target':target};continue
            calibrated[model][name]=match_temperature(value,target)
    # The frozen stopping paragraph says inability to calibrate stops C. Thus
    # every advancing class must match in BOTH directions, without bypassing a
    # failed calibration by selecting another positive class.
    advancing_classes=[name for name,g in gates.items() if g['pass']]
    advance=bool(advancing_classes) and all(calibrated[m][name]['status']=='MATCHED' for name in advancing_classes for m in ['L06','L17'])
    save(run/'B_history_diagnostics.json',{'statistics':stats})
    save(run/'B_gate_and_calibration.json',{'status':'ADVANCE_C' if advance else 'STOP_C','gates':gates,'temperatures':calibrated,'source_ids':sorted({r['sample_id'] for r in rows}),'source_codebook_means':{m:v.tolist() for m,v in source.items()},'all_history_supplemental_source_codebook_means':{m:average(rows,metrics[m],False).tolist() for m in metrics},'protocol_sha':sha(run/'B_protocol.json'),'script_sha':sha(__file__),'no_score_based_calibration':True,'bootstrap_seed':2026102307,'bootstrap_n':20000})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
