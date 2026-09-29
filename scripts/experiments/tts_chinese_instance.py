"""Frozen Chinese within-provider instance experiment; resumable stages."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
from scripts.experiments.tts_independent_visual import ROOT,read,write,sha

BASE=ROOT/'runs'/os.environ.get('TTS_CHINESE_INSTANCE_RUN','tts_chinese_instance_20260926')
SOURCE=ROOT/'runs/aishell1_qwen_mfa_linear_n100_20260816'
CAL=ROOT/'runs/tts_static_lag_calibration_20260926'
PRE=ROOT/'runs/tts_prepost_mel_20260926'
ARMS=('N','C','Q1','Q2')

def freeze():
    import soundfile as sf
    cohort=read(SOURCE/'00_pairs/cohort.json')
    cloud=read(SOURCE/'01_tts_retry/tts_meta.json')
    cal={r['id'] for r in read(CAL/'protocol.json')['rows']}
    rows=[]; excluded=[]
    for r in cohort['records']:
        sid=r['sample_id']; c=cloud['results'][sid]
        try:
            n=Path(r['audio_path']); cp=Path(c['canonical_16k_audio'])
            assert sha(n)==r['audio_sha256']==c['reference_audio_sha256']
            assert sha(cp)==c['canonical_audio_sha256'] and c['transcript']==r['transcript']
            for f in (n,cp):
                info=sf.info(f); assert info.samplerate==16000 and info.channels==1 and info.subtype=='PCM_16'
            rows.append({'id':sid,'speaker':r['speaker_id'],'text':r['transcript'],'split':'calibration' if sid in cal else 'evaluation',
                         'audio':{'N':{'path':str(n),'sha256':sha(n)},'C':{'path':str(cp),'sha256':sha(cp)}},'cloud_voice_id_sha256':hashlib.sha256(c['voice_id'].encode()).hexdigest(),
                         'natural_seconds':sf.info(n).duration})
        except Exception as e: excluded.append({'id':sid,'reason':str(e),'type':type(e).__name__})
    ev=[r for r in rows if r['split']=='evaluation']
    sensitivity=[min(r['id'] for r in ev if r['speaker']==s) for s in sorted({r['speaker'] for r in ev})]
    model=ROOT/'models/Qwen3-TTS-12Hz-0.6B-Base'
    p={'rows':rows,'excluded':excluded,'sensitivity_ids':sensitivity,'images':read(PRE/'protocol.json')['images'],
       'source_sha256':sha(SOURCE/'00_pairs/cohort.json'),'cloud_manifest_sha256':sha(SOURCE/'01_tts_retry/tts_meta.json'),
       'model':str(model),'weights':{str(f):sha(f) for f in sorted(model.rglob('*')) if f.is_file()},
       'local_provider':{'name':'faster_qwen3','strict_backend':True,'dtype':'bfloat16 default','seeds':{'Q1':42,'Q2':43},'max_new_tokens':2048,'language':'Chinese','sampling':{'min_new_tokens':2,'temperature':.9,'top_k':50,'top_p':1.,'do_sample':True,'repetition_penalty':1.05,'xvec_only':False,'non_streaming_mode':False,'append_silence':True}},
       'cloud_provider':{'name':cloud['provider'],'model':cloud['model'],'historical_audio_only':True,'confounds':'checkpoint/provider/enrollment/date differ; not architecture-only contrast'},
       'selection':'all verifiable n100, no prior score/MFA/OOV filtering; old26cal IDs, all remainder eval; IDs not speaker-independent',
       'render':'image3 primary; image6/9 fixed first eval ID per speaker; same static boxes Wav2Lip/FFV1/exact own PCM; reuse only exact compatible cache',
       'lag':'trunc pooled median guard20 best k from independent26cal N/Q1/Q2 image3; arm median range<=1 gate; C separate; all k+-1; secondary faces same primary k with transfer limitation',
       'native':'GQ=(CQ1+CQ2)/2-CN; GC=CC-CN; GQ-GC same available records; guard20 primary,0/15 sensitivity; each instance within utterance then speaker equal',
       'event':'3x3 N/Q1/Q2; fourth C secondary separate support; exact phone sequence, durations>=80ms all arms, phases .25/.5/.75 nearest40ms audio grid center .1075, <=20ms and same interval; visual index j-k guard20; donor common IDs abs displacement3..15 in all arms, >=5 donors/query, >=8queries and>=5 speech occurrences/utterance; no interpolation/warping',
       'primary':'raw Euclidean mean-negative minus positive q; STT=.5(q11+q22-q12-q21); SNT=.25(2qNN+q11+q22-qN1-q1N-qN2-q2N); SNT-STT; raw rank/unit margin/rank and pos/neg decomposition sensitivity',
       'statistics':'speaker equal; 20000 PCG64 bootstrap seed20260926,95/99 CI; no contribution ratios',
       'controls':'first lexicographic ID same input seed42 TTS replay; first eval ID TFG repeat and +/-5frame audio delay all arms; all official matrices replay; audio/mel/MFCC/event differences; no seed replacement; one identical retry for technical exception only',
       'code_sha256_at_freeze':sha(__file__)}
    if (BASE/'protocol.json').exists(): assert read(BASE/'protocol.json')==p
    write(BASE/'protocol.json',p)
    print('frozen',len(rows),'cal',len(cal),'eval',len(ev),'speakers',len({r['speaker'] for r in ev}),'secondary',sensitivity,'duration',sum(r['natural_seconds'] for r in rows),flush=True)

@contextlib.contextmanager
def rng_scope(seed):
    import torch
    py=random.getstate(); ns=np.random.get_state()
    with torch.random.fork_rng(devices=[0]):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        try: yield
        finally: random.setstate(py);np.random.set_state(ns)

def synthesize():
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from scripts.tts.faster_qwen3 import FasterQwen3TTSProvider
    from scripts.experiments.tts_native_gain_attribution.common import gpu_compute_pids,gpu_lease
    p=read(BASE/'protocol.json'); torch.set_num_threads(2)
    for f,h in p['weights'].items(): assert sha(f)==h
    with gpu_lease(gpu_peak_bytes=12<<30,disk_temp_bytes=1<<30,disk_persistent_bytes=1<<30) as gate:
        write(BASE/'tts_compute_gate.json',gate)
        assert not set(gpu_compute_pids())-{os.getpid()}
        provider=FasterQwen3TTSProvider({'faster_qwen3':{'model_id':p['model'],'strict_backend':True,'max_new_tokens':2048}})
        provider._ensure_model()
        if p.get('recovery_mode')=='explicit_warmup_before_seed':
            first=p['rows'][0]
            with rng_scope(20260926):
                provider._model._prepare_generation(text=first['text'],language='Chinese',ref_audio=first['audio']['N']['path'],ref_text=first['text'])
            assert provider._model._warmed_up
            write(BASE/'warmup.json',{'fixed_reference_id':first['id'],'completed_before_generation_seed_scope':True,'warmup_seed':20260926,'dtype':str(provider._model.dtype)})
        import inspect,importlib.metadata
        write(BASE/'tts_runtime.json',{'python':__import__('sys').executable,'torch':torch.__version__,'packages':{k:importlib.metadata.version(k) for k in ('faster-qwen3-tts','qwen-tts','transformers')},'actual_signature':str(inspect.signature(provider._model.generate_voice_clone)),'source_sha256':sha(inspect.getfile(type(provider._model)))})
        jobs=[(r,a,seed) for r in p['rows'] for a,seed in [('Q1',42),('Q2',43)]]+[(p['rows'][0],'Q1_replay',42)]
        for r,a,seed in jobs:
            dest=BASE/'tts'/r['id']/a;receipt=dest.with_suffix('.json')
            if receipt.exists(): continue
            assert not set(gpu_compute_pids())-{os.getpid()}
            errors=[];result=None;start=time.monotonic();termination={}
            def capture_return(frame,event,arg):
                if event=='return':
                    loc=frame.f_locals;steps=len(loc.get('all_codec_ids',[]));token=int(loc['token'].item());limit=loc['talker_graph'].max_seq_len
                    termination.update(steps=steps,token_at_exit=token,eos_id=loc['eos_id'],prefill_len=loc['prefill_len'],max_seq_len=limit,
                        stop_reason='EOS' if token==loc['eos_id'] else 'context_capacity' if loc['prefill_len']+steps>=limit else 'token_limit' if steps>=loc['max_new_tokens'] else 'unresolved',
                        codec_sha256=hashlib.sha256(arg[0].detach().cpu().numpy().tobytes()).hexdigest())
                return capture_return
            def tracer(frame,event,arg):
                if event=='call' and frame.f_code.co_name=='fast_generate':
                    termination['generation_rng_sha256']=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest();return capture_return
            for attempt in range(2):
                try:
                    with rng_scope(seed):
                        if p.get('recovery_mode'):__import__('sys').settrace(tracer)
                        try:result=provider.generate_voice_clone(r['text'],Path(r['audio']['N']['path']),r['text'],'Chinese')
                        finally:__import__('sys').settrace(None)
                    break
                except Exception as e: errors.append({'type':type(e).__name__,'error':str(e)[:500]})
            if result is None:
                write(receipt,{'status':'FAILED','errors':errors});continue
            audio=np.asarray(result.audio,dtype=np.float32)
            if not np.isfinite(audio).all() or len(audio)<100:
                write(receipt,{'status':'DEGENERATE','errors':errors,'samples':len(audio)});continue
            dest.parent.mkdir(exist_ok=True,parents=True)
            raw=dest.with_suffix('.provider.wav');sf.write(raw,audio,result.sample_rate,subtype='PCM_16')
            # Decode the saved provider PCM before the canonical resample, preserving auditability.
            x,sr=sf.read(raw,dtype='float64'); y=resample_poly(x,16000//np.gcd(sr,16000),sr//np.gcd(sr,16000))
            wav=dest.with_suffix('.wav');sf.write(wav,y,16000,subtype='PCM_16')
            write(receipt,{'status':'COMPLETE','id':r['id'],'arm':a,'seed':seed,'seconds':time.monotonic()-start,'duration':len(x)/sr,'peak':float(abs(x).max()),'path':str(wav),'sha256':sha(wav),'provider_path':str(raw),'provider_sha256':sha(raw),'backend_meta':result.backend_meta,'errors':errors,'termination':termination})
            print('tts',r['id'],a,round(len(x)/sr,2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','synthesize'));globals()[parser.parse_args().stage]()
