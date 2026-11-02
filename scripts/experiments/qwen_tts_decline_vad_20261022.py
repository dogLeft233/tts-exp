"""Targeted posthoc VAD sensitivity on all five S0011 conditions; no TFG change."""
from pathlib import Path
import importlib.util
import hashlib
import json
import os
import time
import wave

P=Path(__file__).with_name('qwen_tts_decline_audit_20261022.py')
S=importlib.util.spec_from_file_location('decline',P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)


def main():
    os.environ['HF_HUB_OFFLINE']='1'
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.vad import get_speech_timestamps,VadOptions
    from opencc import OpenCC
    out=M.RUN/'cap_vad_sensitivity';out.mkdir(exist_ok=False);p=M.read(M.RUN/'protocol.json')
    jobs=[j for j in p['jobs'] if j['speaker_id']=='S0011']
    options=dict(threshold=.5,min_speech_duration_ms=100,min_silence_duration_ms=500,speech_pad_ms=200)
    M.save(out/'protocol.json',dict(status='POSTHOC_TARGETED_ASR_DIAGNOSTIC',roles=M.ROLES,options=options,
           source_sha256=M.sha(__file__),purpose='Check speech-bearing tail and no-VAD ASR repetitions; not human proof of spoken words',
           no_audio_edits_for_TFG=True,jobs=jobs))
    model=WhisperModel(p['ASR']['model'],device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True);cc=OpenCC('t2s');rows=[]
    for j in jobs:
        assert M.sha(j['path'])==j['sha256']
        with wave.open(j['path'],'rb') as f:
            assert f.getframerate()==16000 and f.getnchannels()==1 and f.getsampwidth()==2
            pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
        x=pcm.astype(np.float32)/np.float32(32768)
        spans=get_speech_timestamps(x,VadOptions(**options),sampling_rate=16000)
        segments,info=model.transcribe(x,language='zh',beam_size=5,temperature=0,vad_filter=True,vad_parameters=options,condition_on_previous_text=False)
        ss=[dict(start=s.start,end=s.end,text=s.text) for s in segments];raw=''.join(s['text'] for s in ss)
        ref=M.normalize(j['transcript'],cc);hyp=M.normalize(raw,cc)
        row=dict(sample_id=j['sample_id'],role=j['role'],audio_sha256=j['sha256'],spans_s=[dict(start=s['start']/16000,end=s['end']/16000) for s in spans],
                 duration_s=len(x)/16000,VAD_speech_s=sum(s['end']-s['start'] for s in spans)/16000,
                 VAD_after_6s_speech_s=sum(max(0,s['end']-max(s['start'],6*16000)) for s in spans)/16000,
                 raw=raw,reference=ref,normalized=hyp,CER=M.edit_distance(ref,hyp)/max(1,len(ref)),segments=ss)
        M.save(out/(j['role']+'.json'),row);rows.append(row);print('VAD_ASR',j['role'],row['VAD_speech_s'],row['CER'],flush=True)
    M.save(out/'results.json',dict(records=rows,count=len(rows),interpretation='VAD+ASR proxy; no actual human listening, no proof of exact repeated spoken words'))


if __name__=='__main__':main()
