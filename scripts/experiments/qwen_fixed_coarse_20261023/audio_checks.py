"""ASR and MFA timing checks for four fixed-coarse cells, without SyncNet."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_fixed_coarse_20261023_attempt01'
ALIGN=RUN/'checks'
sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_fixed_coarse_20261023.probe import read,save,sha


def timing_ok(labels_equal, deltas, unknown_count, lag_frames, equal_samples):
    values=np.asarray(deltas,dtype=float)
    return bool(labels_equal and len(values) and np.isfinite(values).all() and unknown_count==0 and
        np.median(values)<=20.0001 and np.quantile(values,.95)<=40.0001 and max(values)<=80.0001 and
        abs(lag_frames)<=1 and equal_samples)


def prepare():
    import soundfile as sf
    from scipy.signal import resample_poly
    p=read(RUN/'protocol.json')
    corpus=RUN/'checks/corpus/S0002'
    corpus.mkdir(parents=True,exist_ok=False)
    rows=[]
    for model in ['L06','L17']:
        for stage in ['control','cross']:
            d=RUN/f'{model}_{stage}'
            assert read(d/'result.json')['status']=='PASS'
            source=model if stage=='control' else ('L17' if model=='L06' else 'L06')
            name=f'coarse_{source}_complete_{model}'
            x,sr=sf.read(d/'forced.wav',dtype='float32')
            assert sr==24000
            x=resample_poly(x,2,3).astype('float32')
            gain=.0376838172675746/np.sqrt(np.mean(x.astype(float)**2))
            y=x*np.float32(gain)
            assert max(abs(y))<1
            dest=corpus/(name+'.wav')
            sf.write(dest,y,16000,subtype='PCM_16')
            (corpus/(name+'.lab')).write_text(p['records'][model]['target_text'])
            rows.append(dict(name=name,model=model,coarse_source=source,source_path=str(d/'forced.wav'),
                source_sha256=sha(d/'forced.wav'),path=str(dest),sha256=sha(dest),samples=len(y),gain=float(gain)))
    save(RUN/'checks/audio_manifest.json',dict(records=rows,target_rms=.0376838172675746,
        input_policy='same scipy resample_poly(2,3), whole RMS and PCM16 in all four cells; audio-only technical check'))


def asr():
    import wave
    from faster_whisper import WhisperModel
    from opencc import OpenCC
    from scripts.experiments.qwen_tts_decline_audit_20261022 import normalize,edit_distance
    modeldir=ROOT/'runs/mfa_linear_content_pause_prosody_mapping_20260925/asr_model_small'
    model=WhisperModel(str(modeldir),device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    cc=OpenCC('t2s')
    ref=normalize(read(RUN/'protocol.json')['records']['L06']['target_text'],cc)
    rows=[]
    for r in read(RUN/'checks/audio_manifest.json')['records']:
        assert sha(r['path'])==r['sha256']
        with wave.open(r['path']) as f:
            assert f.getframerate()==16000 and f.getnchannels()==1 and f.getsampwidth()==2
            x=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype('float32')/32768
        segments,info=model.transcribe(x,language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False)
        seg=[dict(start=s.start,end=s.end,text=s.text) for s in segments]
        raw=''.join(s['text'] for s in seg);hyp=normalize(raw,cc)
        row=dict(name=r['name'],text=raw,normalized=hyp,reference=ref,edit_distance=edit_distance(ref,hyp),segments=seg)
        rows.append(row);print('ASR',r['name'],raw,row['edit_distance'],flush=True)
    save(RUN/'checks/asr.json',dict(records=rows,all_exact=all(r['edit_distance']==0 for r in rows),
        model=str(modeldir),model_pins={str(f):sha(f) for f in modeldir.iterdir() if f.is_file()},
        settings=dict(device='cpu',compute_type='int8',language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False,initial_prompt=None),
        limitation='ASR proxy, not listening or perceived naturalness'))


def mfa():
    from scripts.experiments.repeated_take_crossover_20261015.timing_diagnostics import _check_mfa_pins
    pins=_check_mfa_pins()
    exe='/home/wjj/miniconda3/envs/mfa/bin/mfa'
    cmd=[exe,'align',str(ALIGN/'corpus'),'/home/wjj/Documents/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict',
         '/home/wjj/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip',str(ALIGN/'textgrids'),
         '--num_jobs','2','--temporary_directory',str(ALIGN/'mfa_temp'),'--clean']
    env=dict(os.environ);env['PATH']=str(Path(exe).parent)+os.pathsep+env.get('PATH','')
    t=time.monotonic()
    with (ALIGN/'mfa.log').open('x') as f:
        r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    save(ALIGN/'mfa_exit.json',dict(returncode=r.returncode,command=cmd,pins=pins,elapsed_s=time.monotonic()-t))
    if r.returncode: raise RuntimeError('MFA failed')


def summary():
    import soundfile as sf
    from scripts.experiments.repeated_take_crossover_20261015.timing_diagnostics import parse_textgrid,_phone_intervals
    rows=read(RUN/'checks/audio_manifest.json')['records']
    parsed={}
    for r in rows:
        files=list((ALIGN/'textgrids').rglob(r['name']+'.TextGrid'))
        assert len(files)==1
        phones,onset,coverage,labels,unknown,unknown_dur=_phone_intervals(parse_textgrid(files[0]))
        parsed[r['name']]=dict(phones=phones,onset=onset,unknown=unknown,unknown_duration=unknown_dur,
                              grid_sha256=sha(files[0]))
    comparisons=[]
    for source in ['L06','L17']:
        same=next(r for r in rows if r['coarse_source']==source and r['model']==source)
        cross=next(r for r in rows if r['coarse_source']==source and r['model']!=source)
        a,b=parsed[same['name']],parsed[cross['name']]
        labels_equal=[v['label'] for v in a['phones']]==[v['label'] for v in b['phones']]
        events=[]
        if labels_equal:
            for i,(u,v) in enumerate(zip(a['phones'],b['phones'])):
                events.append(dict(index=i,label=u['label'],start_delta_ms=1000*(v['start_s']-u['start_s']),
                    end_delta_ms=1000*(v['end_s']-u['end_s']),
                    center_delta_ms=500*(v['start_s']+v['end_s']-u['start_s']-u['end_s'])))
        deltas=np.abs([r['center_delta_ms'] for r in events])
        x,sr=sf.read(same['path']);y,_=sf.read(cross['path'])
        def env(z):
            n=len(z)//160
            return np.sqrt(np.mean(z[:n*160].reshape(n,160)**2,axis=1))
        ex,ey=env(x),env(y)
        corrs=[]
        for lag in range(-10,11):
            aa=ex[max(0,-lag):min(len(ex),len(ey)-lag)]
            bb=ey[max(0,lag):min(len(ey),len(ex)+lag)]
            corrs.append((lag,float(np.corrcoef(aa,bb)[0,1])))
        lag,corr=max(corrs,key=lambda z:z[1])
        stats=dict(median_abs_ms=float(np.median(deltas)),p95_abs_ms=float(np.quantile(deltas,.95)),max_abs_ms=float(max(deltas))) if len(deltas) else None
        gate=timing_ok(labels_equal,deltas,a['unknown']+b['unknown'],lag,len(x)==len(y))
        comparisons.append(dict(coarse_source=source,labels_equal=labels_equal,phone_count=len(events),
             center=stats,events=events,envelope_best_lag_ms=lag*10,envelope_correlation=corr,
             samples_equal=len(x)==len(y),timing_gate=gate))
    asr_result=read(RUN/'checks/asr.json')
    save(ALIGN/'summary.json',dict(status='PASS' if asr_result['all_exact'] and all(r['timing_gate'] for r in comparisons) else 'HOLD',
         asr_all_exact=asr_result['all_exact'],comparisons=comparisons,parsed=parsed,
         limitation='one sentence, automated boundary estimates; no human timing equivalence, naturalness, or SyncNet claim'))
    print(json.dumps(comparisons,ensure_ascii=False),flush=True)


def lexical_repair():
    """Repair whitespace/OOV input failure only; never choose from alignments."""
    corpus=ALIGN/'corpus/S0002';corpus.mkdir(parents=True,exist_ok=False)
    dictionary=Path('/home/wjj/Documents/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict')
    words={line.split()[0] for line in dictionary.read_text().splitlines() if line.strip()}
    text=read(RUN/'protocol.json')['records']['L06']['target_text']
    tokens=list(text)
    missing=sorted(set(tokens)-words)
    save(ALIGN/'repair_protocol.json',dict(reason='unsegmented sentence treated as one OOV spn in first alignment',
        prior=str(RUN/'checks/summary.json'),policy='one Han character per dictionary token, identical across all four cells',
        text=text,tokens=tokens,missing=missing,dictionary_sha256=sha(dictionary),no_audio_change=True))
    assert not missing, missing
    for r in read(RUN/'checks/audio_manifest.json')['records']:
        os.symlink(r['path'],corpus/(r['name']+'.wav'))
        (corpus/(r['name']+'.lab')).write_text(' '.join(tokens))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','asr','mfa','summary','lexical_repair'])
    ap.add_argument('--lexical-repair',action='store_true')
    args=ap.parse_args()
    if args.lexical_repair: ALIGN=RUN/'checks/mfa_lexical_repair02'
    globals()[args.stage]()
