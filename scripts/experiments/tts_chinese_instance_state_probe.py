"""Fixed prefix and immediate repeat harness for RNG/cache diagnosis."""
import hashlib
import os
import sys
import numpy as np
from scripts.experiments.tts_chinese_instance import BASE,read,write,sha,rng_scope

def main():
    import torch,soundfile as sf
    from scripts.tts.faster_qwen3 import FasterQwen3TTSProvider
    from scripts.experiments.tts_native_gain_attribution.common import gpu_lease,gpu_compute_pids
    p=read(BASE/'protocol.json');rows={r['id']:r for r in p['rows']};out=BASE/'quality/state_probe'
    jobs=[(f'a1_{i:03}',a,seed) for i in range(1,5) for a,seed in [('Q1',42),('Q2',43)]]+[('a1_005','Q1',42),('a1_001','normal_repeat1',42),('a1_001','normal_repeat2',42)]
    write(out/'protocol.json',{'jobs':jobs,'purpose':'fixed original prefix then earliest anomaly; two immediate normal repeats; diagnostic only, no replacement','max_new_tokens':2048,'scope':'same frozen RNG scope/provider; record codec IDs and stop locals without altering sampling','source_sha256':sha(__file__)})
    torch.set_num_threads(2)
    with gpu_lease(gpu_peak_bytes=12<<30,disk_temp_bytes=1<<30,disk_persistent_bytes=1<<30) as gate:
        write(out/'compute_gate.json',gate);assert not set(gpu_compute_pids())-{os.getpid()}
        provider=FasterQwen3TTSProvider({'faster_qwen3':{'model_id':p['model'],'strict_backend':True,'max_new_tokens':2048}});provider._ensure_model()
        results=[]
        for index,(sid,arm,seed) in enumerate(jobs):
            r=rows[sid];dest=out/f'{index:02}_{sid}_{arm}';capture={};codes=[]
            def inner(frame,event,arg):
                if event=='return':
                    d=frame.f_locals;token=int(d['token'].item());steps=len(d['all_codec_ids']);limit=d['talker_graph'].max_seq_len
                    capture.update(steps=steps,token_at_exit=token,eos_id=d['eos_id'],prefill_len=d['prefill_len'],max_seq_len=limit,stop_reason='EOS' if token==d['eos_id'] else 'context_capacity' if d['prefill_len']+steps>=limit else 'token_limit' if steps>=d['max_new_tokens'] else 'unresolved')
                    codes.append(arg[0].detach().cpu().numpy())
                return inner
            def trace(frame,event,arg):return inner if event=='call' and frame.f_code.co_name=='fast_generate' else None
            with rng_scope(seed):
                sys.settrace(trace)
                try:z=provider.generate_voice_clone(r['text'],__import__('pathlib').Path(r['audio']['N']['path']),r['text'],'Chinese')
                finally:sys.settrace(None)
            wav=dest.with_suffix('.wav');sf.write(wav,z.audio,z.sample_rate,subtype='PCM_16');np.savez_compressed(dest.with_suffix('.npz'),codes=codes[0])
            original=BASE/'tts'/sid/('Q1.provider.wav' if seed==42 else 'Q2.provider.wav')
            result={'id':sid,'arm':arm,'seed':seed,'duration':len(z.audio)/z.sample_rate,'same_original_provider_pcm':sha(wav)==sha(original),'sha256':sha(wav),'codec_sha256':hashlib.sha256(codes[0].tobytes()).hexdigest(),'first_codebook_sha256':hashlib.sha256(codes[0][:,0].tobytes()).hexdigest(),**capture}
            write(dest.with_suffix('.json'),result);results.append(result);print(result,flush=True)
        write(out/'results.json',results)

if __name__=='__main__':main()
