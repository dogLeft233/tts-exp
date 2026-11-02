"""Freeze per-policy eligibility before scoring; reuse original official runner."""
from pathlib import Path
import argparse,sys,subprocess
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha
from scripts.experiments.qwen_prediction_mechanism_20261023.budget import guard

def prepare(run):
    p=read(run/'C_protocol.json');addon=read(run/'C_content_addendum.json');assert addon['status']=='FROZEN_BEFORE_C'
    audio={r['id']:r for r in read(run/'C_audio_manifest.json')['records']};asr={r['id']:r for r in read(run/'C_ASR/results.json')['records']}
    byid={r['id']:r for r in p['rows']}
    def canonical_id(key):
        while byid[key]['duplicate_of']:key=byid[key]['duplicate_of']
        return key
    policy=[];jobs=[];reuse=[]
    analysis=run/'C_analysis_protocol.json';save(analysis,{'P1':'L06_both-L06_baseline','P2':'(L17_baseline-L06_baseline)-(L17_both-L06_both)','bootstrap_n':20000,'bootstrap_seed':2026102308,'CI':.975,'units':'10 sources with3draw first averaged','main_all_draws':True,'gate_sha':sha(run/'B_gate_and_calibration.json'),'C_protocol_sha':sha(run/'C_protocol.json')})
    for model in ['L06','L17']:
        for name in ['baseline','coarse','fine','both']:
            rows=[r for r in p['rows'] if r['model']==model and r['policy']==name];bad=[];available=[]
            for row in rows:
                key=canonical_id(row['id']);a=audio[key];s=asr.get(key);reasons=[]
                if a['status']!='COMPLETE':reasons.append('technical_audio_failure')
                elif not a['qualified_EOS_duration']:reasons.append('EOS_or_duration_failure')
                if s is None or s['status']!='COMPLETE':reasons.append('ASR_unavailable')
                else:
                    if addon.get('empty_hypothesis_is_severe',False) and not s['normalized'] and s['reference']:reasons.append('empty_ASR_hypothesis')
                    threshold=addon.get('CER_severe_threshold')
                    if threshold is not None and s['CER']>threshold:reasons.append('predefined_severe_CER')
                if reasons:bad.append({'id':row['id'],'source':row['sample_id'],'reasons':reasons})
                if a['status']=='COMPLETE':available.append((row,a))
            bad_sources=sorted({r['source'] for r in bad});passed=len(bad)<=3 and len(bad_sources)<=2
            summary={'model':model,'policy':name,'bad_draws':len(bad),'draws':30,'bad_sources':bad_sources,'failures':bad,'downstream_eligible':passed,'decision_before_scores':True}
            policy.append(summary)
            if not passed:continue
            # Exact duplicate policies use their unique policy's entire evidence.
            if all(r['duplicate_of'] for r in rows):continue
            if model=='L06' and name=='baseline':
                reuse.extend({'id':r['id'],'score_path':r['old_score'],'score_sha':r['old_score_sha'],'canonical_audio_sha':r['old_audio']['sha256']} for r in rows);continue
            groups={}
            for sample in p['samples']:
                subset=[(r,a) for r,a in available if r['sample_id']==sample['sample_id']]
                roles=tuple('T'+str(r['draw']) for r,a in subset)
                if roles:groups.setdefault(roles,[]).append((sample,subset))
            for i,(roles,group) in enumerate(groups.items()):
                directory=run/f'C_downstream_{model}_{name}_{i}';directory.mkdir();(directory/'receipts').mkdir();(directory/'logs').mkdir()
                records=[];mapping=[]
                for sample,subset in group:
                    for row,a in subset:
                        record=dict(a);record['role']='T'+str(row['draw']);record['cohort']='confirmation_not_calibration';records.append(record);mapping.append({'id':row['id'],'sample_id':row['sample_id'],'role':record['role']})
                config={'samples':[s for s,_ in group],'roles':list(roles),'expected_cells':len(records),'cohort':'confirmation_not_calibration','downstream':p['downstream'],'target_rms':p['target_rms'],'source_pins':p['pins'],'analysis_protocol_path':str(analysis),'execution':{'stage_timeout_s':10800}}
                save(directory/'protocol.json',config);save(directory/'audio_manifest.json',{'records':records,'count':len(records)});save(directory/'unit_mapping.json',mapping)
                jobs.append({'directory':str(directory),'count':len(records),'protocol_sha':sha(directory/'protocol.json'),'model':model,'policy':name})
    save(run/'C_policy_eligibility.json',{'status':'FROZEN_BEFORE_DOWNSTREAM','policies':policy,'rule':'stop policy if >10% of30draws fail or >2sources lack complete qualified support; retain all successful technical audio including small EOS/duration deviations, no score selection','content_addendum_sha':sha(run/'C_content_addendum.json')})
    save(run/'C_downstream_plan.json',{'jobs':jobs,'reuse':reuse,'max_new_video_and_score':sum(j['count'] for j in jobs),'C_protocol_sha':sha(run/'C_protocol.json')})

def execute(run):
    p=read(run/'C_downstream_plan.json');runner=Path(__file__).with_name('downstream_adapter.py');launcher=Path(__file__).with_name('execute.py');py='/home/wjj/.venvs/syncnet/bin/python'
    for job in p['jobs']:
        guard(run)
        directory=Path(job['directory'])
        for stage in ['render','official']:
            cmd=[py,str(runner),'--stage',stage,'--run',str(directory),'--protocol',str(directory/'protocol.json'),'--protocol-sha256',job['protocol_sha']]
            subprocess.run([py,str(launcher),str(directory),stage,*cmd],check=True)
    save(run/'C_downstream_completion.json',{'status':'COMPLETE','jobs':len(p['jobs'])})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','execute']);ap.add_argument('--run',required=True);a=ap.parse_args()
    if a.stage=='prepare':prepare(Path(a.run))
    else:execute(Path(a.run))
