"""Frozen P1/P2 and secondary descriptions on explicit common source support."""
from pathlib import Path
import argparse,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha
from scripts.experiments.qwen_prediction_mechanism_20261023.analyze_bank import entropy
from scripts.experiments.qwen_prediction_mechanism_20261023.cpu_audit import softmax

def describe(x):
    a=np.asarray(x,dtype=float);rng=np.random.default_rng(2026102308)
    if not len(a):return {'n_sources':0,'mean':None,'CI97_5':None}
    b=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)
    return {'n_sources':len(a),'mean':float(a.mean()),'CI97_5':[float(v) for v in np.quantile(b,[.0125,.9875])],'CI95_descriptive':[float(v) for v in np.quantile(b,[.025,.975])],'source_values':a.tolist()}

def main(run):
    p=read(run/'C_protocol.json');plan=read(run/'C_downstream_plan.json');byid={r['id']:r for r in p['rows']};scores={};pins={};unavailable=[]
    for r in plan['reuse']:
        assert sha(r['score_path'])==r['score_sha'];s=read(r['score_path']);assert s['canonical_audio_sha256']==r['canonical_audio_sha'];scores[r['id']]=s;pins[r['score_path']]=r['score_sha']
    for j in plan['jobs']:
        d=Path(j['directory'])
        for r in read(d/'unit_mapping.json'):
            f=d/'official/scores'/f'{r["sample_id"]}_{r["role"]}.json'
            if f.exists():
                s=read(f);pins[str(f)]=sha(f)
                if s['status']=='COMPLETE':scores[r['id']]=s
                else:unavailable.append({'id':r['id'],'status':s['status']})
            else:unavailable.append({'id':r['id'],'status':'NOT_SCORED'})
    def rootkey(key):
        while byid[key]['duplicate_of']:key=byid[key]['duplicate_of']
        return key
    for r in p['rows']:
        key=rootkey(r['id'])
        if key in scores:scores[r['id']]=scores[key]
    samples=[s['sample_id'] for s in p['samples']];means={};entropies=[];primary_support=[]
    required=[('L06','baseline'),('L17','baseline'),('L06','both'),('L17','both')]
    for sid in samples:
        means[sid]={}
        for model in ['L06','L17']:
            for policy in ['baseline','coarse','fine','both']:
                keys=[f'{sid}_{model}_{policy}_{d}' for d in range(1,4)]
                if all(k in scores for k in keys):means[sid][model+'_'+policy]=float(np.mean([scores[k]['sync_c'] for k in keys]))
        if all(m+'_'+pol in means[sid] for m,pol in required):primary_support.append(sid)
    contrasts={}
    for name,fn in [
        ('P1_L06_both_minus_baseline',lambda v:v['L06_both']-v['L06_baseline']),
        ('P2_gap_shrink',lambda v:(v['L17_baseline']-v['L06_baseline'])-(v['L17_both']-v['L06_both'])),
        ('native_model_gap',lambda v:v['L17_baseline']-v['L06_baseline']),
        ('matched_model_gap',lambda v:v['L17_both']-v['L06_both']),
        ('L17_both_minus_baseline',lambda v:v['L17_both']-v['L17_baseline'])]:
        contrasts[name]=describe([fn(means[s]) for s in primary_support])
    secondary=[]
    for model in ['L06','L17']:
        for policy in ['coarse','fine']:
            support=[sid for sid in samples if all(model+'_'+name in means[sid] for name in ['baseline',policy])]
            secondary.append({'model':model,'policy':policy,'source_ids':support,**describe([means[s][model+'_'+policy]-means[s][model+'_baseline'] for s in support])})
        support=[sid for sid in samples if all(model+'_'+name in means[sid] for name in ['baseline','coarse','fine','both'])]
        secondary.append({'model':model,'contrast':'both-coarse-fine+baseline interaction','source_ids':support,**describe([means[s][model+'_both']-means[s][model+'_coarse']-means[s][model+'_fine']+means[s][model+'_baseline'] for s in support])})
    for row in p['rows']:
        if row['duplicate_of']:continue
        unit=run/('C_generation_'+row['model'])/row['id'];f=unit/'trace.npz'
        if not f.exists():continue
        generation_status=read(unit/'result.json')['status']
        if generation_status not in ['COMPLETE','OLD_BASELINE_REUSE']:continue
        z=np.load(f);values=[]
        for cb in range(16):
            e=z[f'effective_{cb}'][:,:2048]
            if cb==0:e=e[:-1]
            values.append(float(entropy(softmax(e)).mean()))
        entropies.append({'id':row['id'],'model':row['model'],'policy':row['policy'],'sample_id':row['sample_id'],'draw':row['draw'],'coarse':values[0],'fine':float(np.mean(values[1:])),'codebooks':values,'provenance':'historical native codes reconstructed through qualified TF' if row['reuse_L06_baseline'] else 'observed actual free generation before processors forcing (none)'})
    # Content/EOS/addendum and actual trajectory manipulation remain required
    # evidence; numerical P1/P2 gates alone never produce a causal success label.
    numeric={}
    for key,threshold in [('P1_L06_both_minus_baseline',.100),('P2_gap_shrink',.200)]:
        c=contrasts[key];numeric[key]=bool(c['mean'] is not None and c['mean']>=threshold and c['CI97_5'][0]>0)
    content=None;strict=None;background=[];manipulation=[]
    audiofile=run/'C_audio_manifest.json';asrfile=run/'C_ASR/results.json'
    if audiofile.exists() and asrfile.exists():
        audio={r['id']:r for r in read(audiofile)['records']};asr={r['id']:r for r in read(asrfile)['records']};contentrows=[];strictvalues=[]
        def qualifies(key):
            key=rootkey(key);a=audio[key];s=asr.get(key)
            return a['status']=='COMPLETE' and a['qualified_EOS_duration'] and s is not None and s['status']=='COMPLETE' and bool(s['normalized'])
        for sid in samples:
            valid=[]
            for draw in range(1,4):
                keys=[f'{sid}_{model}_{policy}_{draw}' for model,policy in required]
                if all(key in scores and qualifies(key) for key in keys):valid.append(draw)
            if valid:
                v={model+'_'+policy:float(np.mean([scores[f'{sid}_{model}_{policy}_{d}']['sync_c'] for d in valid])) for model,policy in required}
                strictvalues.append({'sample_id':sid,'paired_qualified_draw_indices':valid,'P1':v['L06_both']-v['L06_baseline'],'P2':(v['L17_baseline']-v['L06_baseline'])-(v['L17_both']-v['L06_both'])})
            for model in ['L06','L17']:
                for policy in ['baseline','coarse','fine','both']:
                    keys=[f'{sid}_{model}_{policy}_{d}' for d in range(1,4)];ar=[audio[rootkey(k)] for k in keys];sr=[asr.get(rootkey(k)) for k in keys]
                    contentrows.append({'sample_id':sid,'model':model,'policy':policy,'qualified_draw_count':sum(qualifies(k) for k in keys),'technical_audio_failures':sum(a['status']!='COMPLETE' for a in ar),'ASR_CER_per_draw':[s['CER'] if s is not None and s['status']=='COMPLETE' else None for s in sr],'CER_mean_all3':float(np.mean([s['CER'] for s in sr])) if all(s is not None and s['status']=='COMPLETE' for s in sr) else None,'EOS_duration_per_draw':[bool(a['qualified_EOS_duration']) if a['status']=='COMPLETE' else None for a in ar]})
                    if all(k in scores and all(field in scores[k] for field in ['sync_d','av_offset']) for k in keys):background.append({'sample_id':sid,'model':model,'policy':policy,'mean_Dmin':float(np.mean([scores[k]['sync_d'] for k in keys])),'mean_AV_offset_frames':float(np.mean([scores[k]['av_offset'] for k in keys])),'offsets':[scores[k]['av_offset'] for k in keys]})
        content={'status':'ASR_PROXY_ONLY_NO_NEW_CER_REJECTION_THRESHOLD','source_policy_records':contentrows,'empty_ASR_or_unavailable_qualification_separate_from_CER':True}
        strict={'status':'SECONDARY_POSTTREATMENT_CONDITIONAL_NOT_REPLACEMENT_MAIN','paired_draw_support':strictvalues,'P1':describe([r['P1'] for r in strictvalues]),'P2':describe([r['P2'] for r in strictvalues])}
    if (run/'B_gate_and_calibration.json').exists():
        b=read(run/'B_gate_and_calibration.json');ed={r['id']:r for r in entropies}
        for model in ['L06','L17']:
            for category in ['coarse','fine']:
                calibration=b['temperatures'][model][category]
                if calibration['status']!='MATCHED':continue
                values=[];supported=[]
                for sid in samples:
                    keys=[rootkey(f'{sid}_{model}_both_{d}') for d in range(1,4)]
                    if all(k in ed for k in keys):values.append(float(np.mean([ed[k][category] for k in keys])));supported.append(sid)
                achieved=float(np.mean(values)) if values else None;error=achieved-calibration['target'] if achieved is not None else None
                manipulation.append({'model':model,'category':category,'source_ids':supported,'B_frozen_target_nat':calibration['target'],'C_actual_trajectory_mean_nat':achieved,'error_nat':error,'within_original_0_02nat_target':bool(error is not None and abs(error)<=.02),'interpretation':'unmet actual trajectory target is insufficient manipulation, not a negative causal test of entropy'})
    save(run/'C_analysis.json',{'status':'NUMERIC_ANALYSIS_REQUIRES_CONTENT_AND_MANIPULATION_REVIEW','source_ids':samples,'complete_primary_source_ids':primary_support,'missing_primary_source_ids':[s for s in samples if s not in primary_support],'full10_support':len(primary_support)==10,'person_means':means,'primary_and_gap_contrasts':contrasts,'numeric_support_gates':numeric,'secondary':secondary,'actual_trajectory_entropy':entropies,'actual_entropy_target_checks':manipulation,'content_diagnostics':content,'strict_no_anomaly_draw_sensitivity':strict,'Dmin_offset_background':background,'unavailable_scores':unavailable,'score_pins':pins,'C_protocol_sha':sha(run/'C_protocol.json'),'bootstrap_n':20000,'bootstrap_seed':2026102308,'interpretation':'temperature-policy involvement only; not unique entropy mediation, pure scale, visual mouth accuracy, or perceptual quality'})

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
