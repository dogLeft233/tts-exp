"""Inference-only probability observation and full-history forcing.

Forcing occurs AFTER all native processors, and never enters recorded scores.
The original multinomial is retained to preserve RNG consumption.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse, gc, hashlib, inspect, json, math, os, random, signal, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def json_safe(x):
    if isinstance(x,float) and not math.isfinite(x): return str(x)
    if isinstance(x,dict): return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [json_safe(v) for v in x]
    return x
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f: json.dump(json_safe(x),f,ensure_ascii=False,indent=2,allow_nan=False)

@contextmanager
def trace_full(wrapper, forced=None):
    import numpy as np
    import torch
    from transformers import LogitsProcessorList
    talker=wrapper.model.talker; predictor=talker.code_predictor
    original=LogitsProcessorList.__call__; generate=predictor.generate
    state={'frame':-1,'cb':0,'fine':False,'rows':[],'processors':{}}
    def fine_generate(*args,**kwargs):
        state['fine']=True; state['cb']=1
        try: return generate(*args,**kwargs)
        finally: state['fine']=False
    def call(processors,input_ids,scores,**kwargs):
        fine=state['fine']
        if fine: frame=state['frame']; cb=state['cb']; state['cb']+=1
        else: state['frame']+=1; frame=state['frame']; cb=0
        raw=scores.detach().clone()
        effective=original(processors,input_ids,scores,**kwargs)
        state['processors'][str(cb)]=[{'type':type(p).__name__,'state':{k:v for k,v in vars(p).items() if isinstance(v,(int,float,str,bool,list,type(None)))}} for p in processors]
        state['rows'].append(dict(frame=frame,cb=cb,raw=raw[0].cpu().numpy().copy(),
            effective=effective[0].detach().cpu().numpy().copy(),
            probability=torch.softmax(effective,dim=-1)[0].detach().cpu().numpy().copy(),
            history=input_ids[0].detach().cpu().numpy().copy()))
        if forced is None: return effective
        token=int(forced[frame,cb]) if frame<len(forced) else (wrapper.model.config.talker_config.codec_eos_token_id if cb==0 else -1)
        if token<0 or token>=scores.shape[-1]: raise ValueError('history index/token invalid')
        out=effective.new_full(effective.shape,-float('inf')); out[:,token]=0
        return out
    predictor.generate=fine_generate; LogitsProcessorList.__call__=call
    try: yield state
    finally: predictor.generate=generate; LogitsProcessorList.__call__=original

def persist_trace(out,trace):
    import numpy as np
    arrays={}
    for cb in range(16):
        rows=[r for r in trace['rows'] if r['cb']==cb]
        for k in ['raw','effective','probability']: arrays[f'{k}_{cb}']=np.stack([r[k] for r in rows]) if rows else np.empty((0,0),dtype=np.float32)
        arrays[f'frame_{cb}']=np.array([r['frame'] for r in rows])
    np.savez_compressed(out,**arrays)
    save(out.with_suffix('.json'),{'processors':trace['processors'],'rows':[{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in r.items() if k not in ['raw','effective','probability']} for r in trace['rows']]})

def seed(n):
    import numpy as np,torch
    random.seed(n);np.random.seed(n%2**32);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def setup(name,p):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2')
    import torch
    from qwen_tts import Qwen3TTSModel,Qwen3TTSTokenizer
    from scripts.experiments.qwen_prediction_mechanism_20261023.budget import guard
    waited=0
    while True:
        text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        other=[s for s in text.splitlines() if s.strip() and s.split(',')[0].strip()!=str(os.getpid()) and not any(v in s for v in ['gnome-remote-desktop-daemon','/Xorg','gnome-shell'])]
        if not other:break
        guard(ROOT/'runs/qwen_prediction_mechanism_20261023_attempt02')
        if waited%60==0:print(json.dumps({'event':'WAIT_OTHER_GPU_PROCESS','processes':other,'waited_s':waited}),flush=True)
        time.sleep(10);waited+=10
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed(20261022)
    d=p['model_dirs'][name]
    w=Qwen3TTSModel.from_pretrained(d,local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
    w.model.eval().requires_grad_(False)
    old=w.model.speech_tokenizer;w.model.speech_tokenizer=None;del old;gc.collect();torch.cuda.empty_cache()
    c=Qwen3TTSTokenizer.from_pretrained(d+'/speech_tokenizer',local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
    c.model.eval().requires_grad_(False);w.model.speech_tokenizer=c
    return w

def stage_a(name,run):
    import numpy as np, soundfile as sf, torch
    from scripts.experiments.qwen_full_tts_extension_20261022 import tts_stage as h
    start=time.monotonic();out=run/('A_'+name);out.mkdir()
    save(out/'started.json',{'pid':os.getpid(),'unix':time.time(),'command':sys.argv})
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('unit timeout900s')));signal.alarm(900)
    try:
        p=read(ROOT/'runs/qwen_fixed_coarse_20261023_attempt01/protocol.json')
        for f,digest in p['pins'].items(): assert sha(f)==digest,f
        w=setup(name,p);r=p['records'][name]
        save(out/'pins.json',{'old_protocol_sha256':sha(ROOT/'runs/qwen_fixed_coarse_20261023_attempt01/protocol.json'),'verified':p['pins'],'source':sha(__file__),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'model_source':{inspect.getfile(type(m)):sha(inspect.getfile(type(m))) for m in [w,w.model,w.model.talker,w.model.talker.code_predictor]}})
        x,sr=sf.read(p['sample']['input24_path'],dtype='float32');codes=np.load(r['new_codes']['path'])
        with torch.inference_mode(): prompt=w.create_voice_clone_prompt(ref_audio=(x,sr),ref_text=r['target_text'],x_vector_only_mode=False)
        assert np.array_equal(h.codes(prompt[0].ref_code),np.load(r['reference_codes']['path']))
        traces=[];rngs=[]
        for mode in ['native','teacher']:
            seed(p['seed']);t=time.monotonic()
            with torch.inference_mode(),h.capture_official(w) as cap,trace_full(w,codes if mode=='teacher' else None) as trace:
                waves,rate=w.generate_voice_clone(text=r['target_text'],language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**p['generation_kwargs'])
            np.save(out/(mode+'_codes.npy'),cap['new_codes'][0]);h.save_wave(out/(mode+'.wav'),waves[0])
            persist_trace(out/(mode+'.npz'),trace);traces.append(trace)
            rngs.append(torch.cuda.get_rng_state().cpu().numpy().copy())
            result={'codes_exact':bool(np.array_equal(codes,cap['new_codes'][0])),'wave_exact':bool(np.array_equal(waves[0],sf.read(r['path'],dtype='float32')[0])),'elapsed_s':time.monotonic()-t,'trace_rows':len(trace['rows'])}
            save(out/(mode+'_result.json'),result)
            assert result['codes_exact'] and result['wave_exact'],mode+' replay failed'
        assert len(traces[0]['rows'])==len(traces[1]['rows'])
        maxlog=maxprob=maxraw=0.;argmax=True
        expected=[(f,cb) for f in range(len(codes)) for cb in range(16)]+[(len(codes),0)]
        for tr in traces:assert [(r['frame'],r['cb']) for r in tr['rows']]==expected,'frame/codebook order'
        for a,b in zip(traces[0]['rows'],traces[1]['rows']):
            assert (a['frame'],a['cb'])==(b['frame'],b['cb'])
            assert np.isfinite(a['raw']).all() and np.isfinite(b['raw']).all()
            maxraw=max(maxraw,float(np.max(abs(a['raw']-b['raw']))))
            finite=np.isfinite(a['effective']);assert np.array_equal(finite,np.isfinite(b['effective']))
            maxlog=max(maxlog,float(np.max(abs(a['effective'][finite]-b['effective'][finite]))))
            maxprob=max(maxprob,float(np.max(abs(a['probability']-b['probability']))))
            argmax=argmax and bool(a['effective'].argmax()==b['effective'].argmax())
        result={'status':'PASS' if maxraw<=1e-4 and maxlog<=1e-4 and maxprob<=1e-5 and argmax else 'CONTROL_FAILED','max_abs_raw_logits':maxraw,'raw_all_finite':True,'frame_codebook_mapping_exact':True,'expected_trace_rows':len(expected),'max_abs_effective_logits':maxlog,'max_abs_probability':maxprob,'argmax_exact':argmax,'rng_exact':bool(np.array_equal(*rngs)),'elapsed_s':time.monotonic()-start}
        save(out/'result.json',result);assert result['status']=='PASS' and result['rng_exact']
    except BaseException as exc:
        save(out/'failure.json',{'status':'CONTROL_FAILED','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-start});raise
    finally: signal.alarm(0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    stage_a(a.model,Path(a.run))
