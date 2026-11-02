"""Post-hoc decline/influence and independent ASR diagnostics; no TFG or training."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/qwen_full_tts_extension_20261022_attempt01'
RUN=ROOT/'runs/qwen_tts_decline_audit_20261022_attempt01'
MODEL=ROOT/'runs/mfa_linear_content_pause_prosody_mapping_20260925/asr_model_small'
ROLES=['N','R','T1','T2','T3']
POLICIES=['common','native_full','equal_count_spread']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,c in enumerate(a,1):
        new=[i]
        for j,d in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(c!=d)))
        row=new
    return row[-1]


def normalize(text,converter):
    return ''.join(c for c in converter.convert(unicodedata.normalize('NFKC',text).lower())
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def prepare():
    p=read(BASE/'protocol.json');parent=Path(p['parent_run'])
    t=read(BASE/'tts/tts_manifest.json')['records']
    records=read(parent/'audio_manifest.json')['records']+read(BASE/'audio_manifest.json')['records']
    samples={s['sample_id']:s for s in p['samples']}
    jobs=[]
    for s in p['samples']:
        for role in ROLES:
            row=next(r for r in records if r['sample_id']==s['sample_id'] and r['role']==role)
            assert sha(row['path'])==row['sha256']
            item=dict(sample_id=s['sample_id'],speaker_id=s['speaker_id'],role=role,transcript=s['transcript'],
                      path=row['path'],sha256=row['sha256'],natural_duration_s=s['duration_s'])
            if role.startswith('T'):
                rr=next(r for r in t if r['sample_id']==s['sample_id'] and r['seed_index']==int(role[1:]))
                item.update(seed=rr['seed'],EOS=rr['eos_in_talker_sequences'],new_steps=rr['new_steps'])
            jobs.append(item)
    model_files={str(f):sha(f) for f in MODEL.iterdir() if f.is_file()}
    save(RUN/'protocol.json',dict(status='POSTHOC_DIAGNOSTIC_NOT_PRIMARY_REANALYSIS',jobs=jobs,
         source_analysis=dict(path=str(BASE/'analysis.json'),sha256=sha(BASE/'analysis.json')),
         source_seal=dict(path=str(BASE/'artifact_manifest.json'),sha256=sha(BASE/'artifact_manifest.json')),
         ASR=dict(model=str(MODEL),model_files=model_files,device='cpu',compute_type='int8',cpu_threads=4,
                  language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False,initial_prompt=None,
                  normalization='NFKC lowercase, OpenCC t2s, remove spaces and Unicode punctuation; number spelling differences retained',
                  limitation='ASR proxy, no human listening or MOS; CER is not mouth quality'),
         scope='all 50 canonical N/R/T1/T2/T3 audios; no score-based omission; all original results sealed and unchanged',
         diagnostics=['all-speaker leave-one-out means','remove S0010 and S0011 jointly, posthoc selected, descriptive only',
                      'omit capped draw only, keep original support and other draws, descriptive only',
                      'duration ratio, fixed-frame quiet fractions, transcript and independent ASR character errors'],
         source_sha256=sha(__file__)))
    print('PREPARED',len(jobs),flush=True)


def statistics():
    import numpy as np
    p=read(RUN/'protocol.json');src=p['source_analysis'];assert sha(src['path'])==src['sha256'];a=read(src['path'])
    result={}
    for policy in POLICIES:
        rows=[]
        for r in a['records']:
            c=r['policies'][policy];t=float(np.mean([c[k]['C'] for k in ROLES[2:]]))
            rows.append(dict(sample_id=r['sample_id'],speaker_id=r['speaker_id'],N=c['N']['C'],R=c['R']['C'],T=t,
                             T_minus_N=t-c['N']['C'],T_minus_R=t-c['R']['C'],common_queries=r['common_query_count'],
                             draws=[dict(role=k,C=c[k]['C'],T_minus_N=c[k]['C']-c['N']['C'],T_minus_R=c[k]['C']-c['R']['C']) for k in ROLES[2:]]))
        summaries={}
        for key in ['T_minus_N','T_minus_R']:
            x=np.array([r[key] for r in rows]);assert np.allclose(x,a['summary'][policy]['C']['contrasts'][key.replace('_minus_','-')]['values'],rtol=0,atol=1e-12)
            loo=[dict(omitted=rows[i]['sample_id'],remaining_mean=float(np.delete(x,i).mean()),change_from_full=float(np.delete(x,i).mean()-x.mean())) for i in range(10)]
            remaining=[r[key] for r in rows if r['speaker_id'] not in ['S0010','S0011']]
            summaries[key]=dict(n=10,mean=float(x.mean()),median=float(np.median(x)),positive=int((x>0).sum()),
                                positive_sum=float(x[x>0].sum()),negative_sum=float(x[x<0].sum()),leave_one_out=loo,
                                posthoc_without_S0010_S0011=dict(n=8,mean=float(np.mean(remaining)),not_confirmatory=True))
        last=next(r for r in rows if r['speaker_id']=='S0011')
        before=last['T'];after=float(np.mean([d['C'] for d in last['draws'][:2]]))
        cap=dict(speaker='S0011',role='T3',before_T_mean=before,after_T_mean=after,
                 T_minus_N_full_mean_after_omitting_capped_draw=summaries['T_minus_N']['mean']+(after-before)/10,
                 T_minus_R_full_mean_after_omitting_capped_draw=summaries['T_minus_R']['mean']+(after-before)/10,
                 support='original support fixed; no recomputation/selection',not_confirmatory=True)
        result[policy]=dict(rows=rows,summaries=summaries,capped_draw_influence=cap)
    save(RUN/'influence.json',result);print('STATS_DONE',flush=True)


def audio():
    import numpy as np
    import soundfile as sf
    p=read(RUN/'protocol.json');rows=[]
    for j in p['jobs']:
        assert sha(j['path'])==j['sha256'];x,sr=sf.read(j['path'],dtype='float64');assert sr==16000 and x.ndim==1
        frames=np.lib.stride_tricks.sliding_window_view(x,320)[::160];rms=np.sqrt(np.mean(frames*frames,axis=1))
        row=dict(**j,duration_s=len(x)/sr,duration_ratio_to_natural=(len(x)/sr)/j['natural_duration_s'],
                 rms=float(np.sqrt(np.mean(x*x))),peak=float(abs(x).max()),saturation_count=int((abs(x)>=.999969).sum()),
                 quiet_fraction_below_minus40dBFS=float((rms<.01).mean()),quiet_fraction_below_minus50dBFS=float((rms<10**(-50/20)).mean()),
                 characters_per_total_second=len(''.join(j['transcript'].split()))/(len(x)/sr),
                 frame_definition='20ms RMS/10ms hop; quiet proxy is not phonetic/VAD ground truth')
        rows.append(row)
    save(RUN/'audio_diagnostics.json',dict(count=len(rows),records=rows));print('AUDIO_DIAGNOSTICS_DONE',flush=True)


def asr():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from faster_whisper import WhisperModel
    from opencc import OpenCC
    p=read(RUN/'protocol.json');settings=p['ASR']
    for file,h in settings['model_files'].items():assert sha(file)==h
    save(RUN/'asr_runtime.json',dict(packages={n:importlib.metadata.version(n) for n in ['faster-whisper','ctranslate2','opencc-python-reimplemented','numpy']},source_sha256=sha(__file__),no_training=True,device='cpu'))
    def timeout_handler(*_):raise TimeoutError('ASR 1200 second hard limit')
    signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(1200)
    start=time.monotonic();cc=OpenCC('t2s');model=WhisperModel(settings['model'],device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    records=[]
    for j in p['jobs']:
        assert sha(j['path'])==j['sha256']
        segments,meta=model.transcribe(j['path'],language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False)
        ss=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments]
        raw=''.join(s['text'] for s in ss);ref=normalize(j['transcript'],cc);hyp=normalize(raw,cc);distance=edit_distance(ref,hyp)
        row=dict(sample_id=j['sample_id'],role=j['role'],audio_sha256=j['sha256'],raw=raw,reference=ref,normalized=hyp,
                 edit_distance=distance,reference_characters=len(ref),CER=distance/max(1,len(ref)),segments=ss,duration_s=meta.duration)
        save(RUN/'asr'/j['sample_id']/(j['role']+'.json'),row);records.append(row)
        print('ASR_DONE',len(records),j['sample_id'],j['role'],'CER',round(row['CER'],3),flush=True)
    save(RUN/'asr_results.json',dict(count=len(records),records=records,elapsed_s=time.monotonic()-start,verification='ASR proxy, not human verified'))
    signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','statistics','audio','asr'])
    globals()[parser.parse_args().stage]()
