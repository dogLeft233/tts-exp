"""Cold-process diagnostic replay only; never replaces the frozen cohort."""
import argparse
import hashlib
import os
import sys
import time
import numpy as np
from scripts.experiments.tts_chinese_instance import BASE,ROOT,read,write,sha,rng_scope

def main(sid,tag=''):
    import torch,soundfile as sf
    from scipy.signal import resample_poly
    from scripts.tts.faster_qwen3 import FasterQwen3TTSProvider
    from scripts.experiments.tts_native_gain_attribution.common import gpu_lease,gpu_compute_pids
    p=read(BASE/'protocol.json');r=next(r for r in p['rows'] if r['id']==sid)
    out=BASE/'quality/diagnostics'/(sid+tag);out.mkdir(exist_ok=True,parents=True)
    write(out/'protocol.json',{'purpose':'cold process fixed same-input seed42 diagnostic, no replacement','id':sid,'seed':42,'max_new_tokens':2048,'protocol_sha256':sha(BASE/'protocol.json'),'script_sha256':sha(__file__),'selection':'a1_005 first >30-second record; a1_001 lexicographically first normal control','trace':'inspect fast_generate return locals: token/eos, steps/prefill/maxseq; no RNG changes'})
    torch.set_num_threads(2)
    with gpu_lease(gpu_peak_bytes=12<<30,disk_temp_bytes=1<<30,disk_persistent_bytes=1<<30) as gate:
        write(out/'compute_gate.json',gate);assert not set(gpu_compute_pids())-{os.getpid()}
        provider=FasterQwen3TTSProvider({'faster_qwen3':{'model_id':p['model'],'strict_backend':True,'max_new_tokens':2048}});provider._ensure_model()
        write(out/'actual_model.json',{'class':str(type(provider._model)),'dtype':str(provider._model.dtype),'model_parameter_dtype':str(next(provider._model.model.model.parameters()).dtype)})
        captures=[]
        def inner(frame,event,arg):
            if event=='return':
                loc=frame.f_locals;token=loc.get('token');graph=loc.get('talker_graph')
                captures.append({'steps':len(loc.get('all_codec_ids',[])),'token_at_exit':int(token.item()) if token is not None else None,'eos_id':loc.get('eos_id'),'prefill_len':loc.get('prefill_len'),'step_idx':loc.get('step_idx'),'max_new_tokens':loc.get('max_new_tokens'),'max_seq_len':getattr(graph,'max_seq_len',None),'source_sha256':sha(frame.f_code.co_filename)})
            return inner
        def trace(frame,event,arg):
            return inner if event=='call' and frame.f_code.co_name=='fast_generate' else None
        start=time.monotonic()
        with rng_scope(42):
            sys.settrace(trace)
            try:result=provider.generate_voice_clone(r['text'],__import__('pathlib').Path(r['audio']['N']['path']),r['text'],'Chinese')
            finally:sys.settrace(None)
        raw=out/'cold.provider.wav';sf.write(raw,np.asarray(result.audio),result.sample_rate,subtype='PCM_16')
        x,sr=sf.read(raw,dtype='float64');y=resample_poly(x,16000//np.gcd(sr,16000),sr//np.gcd(sr,16000));sf.write(out/'cold.wav',y,16000,subtype='PCM_16')
        original=BASE/'tts'/sid/'Q1.provider.wav';o,_=sf.read(original,dtype='float64');n=min(len(o),len(x))
        for c in captures:
            c['stop_reason']='EOS' if c['token_at_exit']==c['eos_id'] else ('context_capacity' if c['prefill_len']+c['steps']>=c['max_seq_len'] else 'token_limit' if c['steps']>=c['max_new_tokens'] else 'unresolved')
        z={'seconds':time.monotonic()-start,'duration':len(x)/sr,'original_duration':len(o)/sr,'sha_equal':sha(raw)==sha(original),'sha256':sha(raw),'common_sample_rmse':float(np.sqrt(np.mean((x[:n]-o[:n])**2))),'captures':captures,'anomaly_duration_gt3natural':len(x)/sr>3*r['natural_seconds']}
        write(out/'result.json',z);print(z,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('id',choices=('a1_005','a1_001'));p.add_argument('--tag',default='');args=p.parse_args();main(args.id,args.tag)
