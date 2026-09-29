"""One-factor diagnostic: move lazy graph initialization before generation seed."""
import argparse
import hashlib
import os
import sys
import numpy as np
from pathlib import Path
from scripts.experiments.tts_chinese_instance import BASE,read,write,sha,rng_scope

def main(tag):
    import torch,soundfile as sf
    from scripts.tts.faster_qwen3 import FasterQwen3TTSProvider
    from scripts.experiments.tts_native_gain_attribution.common import gpu_lease,gpu_compute_pids
    p=read(BASE/'protocol.json');rows={r['id']:r for r in p['rows']};out=BASE/'quality'/('rng_probe_'+tag)
    jobs=[('a1_001',42),('a1_001',42),('a1_005',42),('a1_005',42)]
    write(out/'protocol.json',{'jobs':jobs,'change':'initialize CUDA graphs using fixed first reference before per-call seed; no sampler/ICL/checkpoint changes','purpose':'diagnostic reproducibility; not cohort replacement','source_sha256':sha(__file__)})
    torch.set_num_threads(2)
    with gpu_lease(gpu_peak_bytes=12<<30,disk_temp_bytes=1<<30,disk_persistent_bytes=1<<30) as gate:
        write(out/'compute_gate.json',gate);assert not set(gpu_compute_pids())-{os.getpid()}
        provider=FasterQwen3TTSProvider({'faster_qwen3':{'model_id':p['model'],'strict_backend':True,'max_new_tokens':2048}});provider._ensure_model()
        r=rows['a1_001']
        with rng_scope(20260926):
            before=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
            provider._model._prepare_generation(text=r['text'],language='Chinese',ref_audio=r['audio']['N']['path'],ref_text=r['text'])
            after=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        write(out/'warmup.json',{'cuda_rng_before':before,'cuda_rng_after':after,'rng_changed':before!=after,'warmed_up':provider._model._warmed_up,'dtype':str(provider._model.dtype)})
        results=[]
        for i,(sid,seed) in enumerate(jobs):
            r=rows[sid];capture={};dest=out/f'{i}_{sid}'
            def inner(frame,event,arg):
                if event=='return':
                    d=frame.f_locals;steps=len(d['all_codec_ids']);token=int(d['token'].item());limit=d['talker_graph'].max_seq_len
                    codes=arg[0].detach().cpu().numpy();np.savez_compressed(dest.with_suffix('.npz'),codes=codes)
                    capture.update(steps=steps,prefill=d['prefill_len'],stop='EOS' if token==d['eos_id'] else 'context_capacity' if steps+d['prefill_len']>=limit else 'other',codec_sha256=hashlib.sha256(codes.tobytes()).hexdigest())
                return inner
            def trace(frame,event,arg):
                if event=='call' and frame.f_code.co_name=='fast_generate':
                    capture['generation_rng_sha256']=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest();return inner
            with rng_scope(seed):
                sys.settrace(trace)
                try:z=provider.generate_voice_clone(r['text'],Path(r['audio']['N']['path']),r['text'],'Chinese')
                finally:sys.settrace(None)
            sf.write(dest.with_suffix('.wav'),z.audio,z.sample_rate,subtype='PCM_16')
            capture.update(id=sid,duration=len(z.audio)/z.sample_rate,sha256=sha(dest.with_suffix('.wav')))
            results.append(capture);write(dest.with_suffix('.json'),capture);print(capture,flush=True)
        write(out/'results.json',results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('tag');main(p.parse_args().tag)
