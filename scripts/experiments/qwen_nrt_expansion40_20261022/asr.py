"""CPU-only ASR audit of all 150 new canonical audios; no selection by score."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import time
import wave

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_tts_decline_audit_20261022 import normalize,edit_distance
RUN=ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01'
MODEL=ROOT/'runs/mfa_linear_content_pause_prosody_mapping_20260925/asr_model_small'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.vad import get_speech_timestamps,VadOptions
    from opencc import OpenCC
    p=read(RUN/'protocol.json');samples={s['sample_id']:s for s in p['samples'] if s['cohort']=='new30'}
    rows=[r for r in read(RUN/'audio_manifest.json')['records'] if not r['reused']];assert len(rows)==150
    out=RUN/'asr';out.mkdir(exist_ok=False)
    model_files={str(f):sha(f) for f in MODEL.iterdir() if f.is_file()}
    old=read(ROOT/'runs/qwen_tts_decline_audit_20261022_attempt01/protocol.json')['ASR']['model_files']
    assert model_files==old
    save(out/'runtime.json',dict(model=str(MODEL),model_files=model_files,source_sha256=sha(Path(__file__)),device='cpu',compute_type='int8',
        packages={n:importlib.metadata.version(n) for n in ['faster-whisper','ctranslate2','numpy','opencc-python-reimplemented']},
        settings=dict(language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False,initial_prompt=None),
        input='exact mono16k signed16 PCM converted float32/32768; bypass AV; all150 retained',
        limitations='ASR proxy not human truth, naturalness or visual accuracy; Arabic/Han number spellings retained in CER'))
    def timeout(*_):raise TimeoutError('ASR1800s hard limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    start=time.monotonic();cc=OpenCC('t2s');model=WhisperModel(str(MODEL),device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    def pcm(j):
        path=Path(j['path']);assert sha(path)==j['sha256']
        with wave.open(str(path)) as w:
            assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2
            raw=w.readframes(w.getnframes())
        assert hashlib.sha256(raw).hexdigest()==j['pcm_sha256']
        return np.frombuffer(raw,dtype='<i2').astype(np.float32)/np.float32(32768)
    result=[]
    for j in rows:
        x=pcm(j);segments,info=model.transcribe(x,language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False)
        ss=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments]
        raw=''.join(s['text'] for s in ss);ref=normalize(samples[j['sample_id']]['transcript'],cc);hyp=normalize(raw,cc);d=edit_distance(ref,hyp)
        row=dict(sample_id=j['sample_id'],speaker_id=j['speaker_id'],role=j['role'],audio_sha256=j['sha256'],pcm_sha256=j['pcm_sha256'],
                 raw=raw,reference=ref,normalized=hyp,edit_distance=d,CER=d/len(ref),segments=ss,duration_s=info.duration)
        save(out/'records'/j['sample_id']/(j['role']+'.json'),row);result.append(row)
        print('ASR',len(result),'/150',j['sample_id'],j['role'],round(row['CER'],3),flush=True)
    save(out/'results.json',dict(count=150,records=result,elapsed_s=time.monotonic()-start))
    # Metadata-triggered follow-up is exploratory; never controls scoring/inclusion.
    t=read(RUN/'generation/tts_manifest.json')['records'];bad=set()
    for r in t:
        ratio=r['samples']/r['sample_rate_hz']/samples[r['sample_id']]['duration_s']
        if r['eos_in_talker_sequences'] is not True or not .5<=ratio<=2:bad.add(r['sample_id'])
    opts=dict(threshold=.5,min_speech_duration_ms=100,min_silence_duration_ms=500,speech_pad_ms=200)
    save(out/'vad_followup_protocol.json',dict(status='EXPLORATORY_METADATA_TRIGGERED_NOT_MAIN_QUALITY_GATE',speakers=sorted(bad),options=opts,
           rule='all five conditions for speakers with any nonEOS or duration-ratio outside .5..2 draw; no score selection'))
    vr=[]
    for j in [r for r in rows if r['sample_id'] in bad]:
        x=pcm(j);spans=get_speech_timestamps(x,VadOptions(**opts),sampling_rate=16000)
        ss,info=model.transcribe(x,language='zh',beam_size=5,temperature=0,vad_filter=True,vad_parameters=opts,condition_on_previous_text=False)
        segments=[dict(start=s.start,end=s.end,text=s.text) for s in ss];text=''.join(s['text'] for s in segments)
        ref=normalize(samples[j['sample_id']]['transcript'],cc);hyp=normalize(text,cc)
        row=dict(sample_id=j['sample_id'],role=j['role'],audio_sha256=j['sha256'],raw=text,normalized=hyp,reference=ref,CER=edit_distance(ref,hyp)/len(ref),
                 spans_s=[dict(start=s['start']/16000,end=s['end']/16000) for s in spans],speech_s=sum(s['end']-s['start'] for s in spans)/16000,segments=segments)
        save(out/'vad'/j['sample_id']/(j['role']+'.json'),row);vr.append(row);print('VAD_ASR',j['sample_id'],j['role'],flush=True)
    save(out/'vad_results.json',dict(count=len(vr),records=vr,human_verified=False))
    signal.alarm(0)


if __name__=='__main__':main()
