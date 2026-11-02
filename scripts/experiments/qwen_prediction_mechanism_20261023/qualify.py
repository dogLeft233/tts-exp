"""Causal ordering tests on short, complete-code forced prefixes; no scoring."""
from pathlib import Path
import argparse, os, signal, sys, time, traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,setup,trace_full,seed,persist_trace

def main(a):
    import numpy as np,soundfile as sf,torch
    from scripts.experiments.qwen_full_tts_extension_20261022 import tts_stage as h
    run=Path(a.run);out=run/('A_qualification_'+a.model);out.mkdir()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('qualification900s')));signal.alarm(900)
    save(out/'started.json',{'pid':os.getpid(),'command':sys.argv,'unix':time.time()})
    try:
        p=read(ROOT/'runs/qwen_fixed_coarse_20261023_attempt01/protocol.json');w=setup(a.model,p);r=p['records'][a.model]
        x,sr=sf.read(p['sample']['input24_path'],dtype='float32')
        with torch.inference_mode():prompt=w.create_voice_clone_prompt(ref_audio=(x,sr),ref_text=r['target_text'],x_vector_only_mode=False)
        codes=np.load(r['new_codes']['path'])[:4].copy();traces=[]
        for name in ['prefix','future','past']:
            forced=codes.copy()
            if name=='future':forced[3,7]=(forced[3,7]+1)%2048
            if name=='past':forced[1,7]=(forced[1,7]+1)%2048
            seed(p['seed'])
            with torch.inference_mode(),h.capture_official(w) as cap,trace_full(w,forced) as tr:
                w.generate_voice_clone(text=r['target_text'],language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**p['generation_kwargs'])
            assert np.array_equal(cap['new_codes'][0],forced),'hidden resampling'
            persist_trace(out/(name+'.npz'),tr);traces.append(tr)
        base,future,past=traces
        by=lambda t:{(r['frame'],r['cb']):r for r in t['rows']}
        b,f,past=map(by,[base,future,past])
        future_exact=all(np.array_equal(r['raw'],f[k]['raw']) for k,r in b.items() if k<=(3,7))
        past_prefix_exact=all(np.array_equal(r['raw'],past[k]['raw']) for k,r in b.items() if k<=(1,7))
        fine_changed=not np.array_equal(b[(1,8)]['raw'],past[(1,8)]['raw'])
        main_changed=not np.array_equal(b[(2,0)]['raw'],past[(2,0)]['raw'])
        result={'status':'PASS' if all([future_exact,past_prefix_exact,fine_changed,main_changed]) else 'CONTROL_FAILED','future_prediction_prefix_exact':future_exact,'past_prediction_prefix_exact':past_prefix_exact,'past_fine_token_changes_following_fine_prediction':fine_changed,'past_fine_token_changes_next_main_prediction':main_changed,'forced_codes_exact_all_three':True,'no_native_generation':True}
        save(out/'result.json',result);assert result['status']=='PASS'
    except BaseException as e:save(out/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--run',required=True);main(ap.parse_args())
