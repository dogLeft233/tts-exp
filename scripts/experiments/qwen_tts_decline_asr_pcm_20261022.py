"""ASR engineering attempt02: exact mono16k PCM input bypasses incompatible PyAV."""
from pathlib import Path
import importlib.util
import importlib.metadata
import json
import os
import signal
import time
import wave
import hashlib

SOURCE=Path(__file__).with_name('qwen_tts_decline_audit_20261022.py')
spec=importlib.util.spec_from_file_location('decline',SOURCE);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    from faster_whisper import WhisperModel
    from opencc import OpenCC
    p=M.read(M.RUN/'protocol.json');out=M.RUN/'asr_attempt02';out.mkdir(exist_ok=False)
    settings=p['ASR']
    for path,h in settings['model_files'].items():assert M.sha(path)==h
    M.save(out/'runtime.json',dict(source_sha256=M.sha(__file__),helper_source_sha256=M.sha(SOURCE),
           packages={n:importlib.metadata.version(n) for n in ['faster-whisper','ctranslate2','opencc-python-reimplemented','numpy','av']},
           input='wave module reads exact canonical mono16k signed16 PCM; float32/32768 ndarray, no resample',
           failed_attempt='asr_console.log: incompatible PyAV metadata_errors argument, zero ASR records',device='cpu',no_training=True))
    def timeout(*_):raise TimeoutError('ASR hard limit1200s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    start=time.monotonic();cc=OpenCC('t2s');model=WhisperModel(settings['model'],device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    records=[]
    for j in p['jobs']:
        assert M.sha(j['path'])==j['sha256']
        with wave.open(j['path'],'rb') as f:
            assert f.getnchannels()==1 and f.getsampwidth()==2 and f.getframerate()==16000
            pcm=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
        x=pcm.astype(np.float32)/np.float32(32768)
        segments,meta=model.transcribe(x,language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False)
        ss=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments]
        raw=''.join(s['text'] for s in ss);ref=M.normalize(j['transcript'],cc);hyp=M.normalize(raw,cc);d=M.edit_distance(ref,hyp)
        row=dict(sample_id=j['sample_id'],role=j['role'],audio_sha256=j['sha256'],pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                 raw=raw,reference=ref,normalized=hyp,edit_distance=d,reference_characters=len(ref),CER=d/max(1,len(ref)),segments=ss,duration_s=meta.duration)
        M.save(out/'asr'/j['sample_id']/(j['role']+'.json'),row);records.append(row)
        print('ASR_DONE',len(records),j['sample_id'],j['role'],'CER',round(row['CER'],3),flush=True)
    M.save(out/'results.json',dict(count=len(records),records=records,elapsed_s=time.monotonic()-start,verification='ASR proxy, not human verified'))
    signal.alarm(0)


if __name__=='__main__':main()
