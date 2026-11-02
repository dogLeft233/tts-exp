"""Frozen common-history bank, inference-only 70 histories x two readers."""
from pathlib import Path
import argparse,json,os,sys,time,signal,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha,setup,seed,trace_full,persist_trace
from scripts.experiments.qwen_prediction_mechanism_20261023.budget import guard

def prepare(run):
    p=read(ROOT/'runs/qwen17_level_comparison10_20261023_attempt01/protocol.json')
    old=read(ROOT/'runs/qwen17_level_comparison10_20261023_attempt02/existing_sources.json')['records']
    qual=read(ROOT/'runs/qwen17_level_comparison10_20261023_attempt02/reliability_sensitivity.json')['qualifications']
    lookup={(r['sample_id'],r['model'],r['draw_index']):r for r in qual}
    rows=[];pins={}
    for s in p['samples']:
        sid=s['sample_id'];refs=[]
        for model in ['L06','L17']:
            for draw in range(1,4):
                if model=='L06':
                    src=next(r for r in old if r['sample_id']==sid and r['model']==model and r['draw_index']==draw)
                    path=Path(src['source_path']).parent/'record.json'
                else:path=ROOT/f'runs/qwen17_level_comparison10_20261023_attempt01/generation/{sid}/draw{draw}.json'
                r=read(path);pins[str(path)]=sha(path)
                for key in ['new_codes','reference_codes','talker_sequences']:
                    a=r[key];assert sha(a['path'])==a['sha256'];pins[a['path']]=a['sha256']
                refs.append(r['reference_codes']['sha256'])
                rows.append({'id':f'{sid}_{model}_{draw}','sample_id':sid,'speaker_id':s['speaker_id'],'source':model,'draw':draw,'seed':r['seed'],'codes':r['new_codes'],'record':str(path),'qualified':lookup[(sid,model,draw)]['qualified'],'old_qualification':lookup[(sid,model,draw)],'terminal_EOS_observed':r['eos_in_talker_sequences']})
        assert len(set(refs))==1
        rows.append({'id':f'{sid}_N_0','sample_id':sid,'speaker_id':s['speaker_id'],'source':'N','draw':0,'seed':20261022,'codes':r['reference_codes'],'qualified':True,'terminal_EOS_observed':False})
        assert sha(s['input24_path'])==s['input24_sha256'];pins[s['input24_path']]=s['input24_sha256']
    assert len(rows)==70
    save(run/'B_protocol.json',{'rows':rows,'samples':p['samples'],'pins':pins,'generation_kwargs':p['generation_kwargs'],'weighting':'source equal; N/T06/T17 equal within source; draw average within T; content frames equal; codebooks individually and fine equal average','qualification_source':'qwen17_level_comparison10_20261023_attempt02/reliability_sensitivity.json','terminal_policy':'append EOS only to terminate evaluation; content predictions before force; N terminal EOS is synthetic and not observed NLL evidence; long/incomplete original histories supplemental only'})

def run_bank(name,run):
    import numpy as np,soundfile as sf,torch
    from scripts.experiments.qwen_full_tts_extension_20261022 import tts_stage as h
    for model in ['L06','L17']:
        assert read(run/f'A_{model}/result.json')['status']=='PASS'
        assert read(run/f'A_qualification_{model}/result.json')['status']=='PASS'
    assert read(run/'A_CPU_independent_audit.json')['status']=='PASS'
    p=read(run/'B_protocol.json');oldp=read(ROOT/'runs/qwen_fixed_coarse_20261023_attempt01/protocol.json')
    out=run/('B_'+name);out.mkdir();save(out/'started.json',{'pid':os.getpid(),'unix':time.time(),'protocol_sha':sha(run/'B_protocol.json')})
    for path,digest in p['pins'].items():assert sha(path)==digest,path
    w=setup(name,oldp);completed=[]
    try:
        for sample in p['samples']:
            sid=sample['sample_id'];text=''.join(sample['transcript'].split());x,sr=sf.read(sample['input24_path'],dtype='float32')
            with torch.inference_mode():prompt=w.create_voice_clone_prompt(ref_audio=(x,sr),ref_text=text,x_vector_only_mode=False)
            n=next(r for r in p['rows'] if r['sample_id']==sid and r['source']=='N')
            assert np.array_equal(h.codes(prompt[0].ref_code),np.load(n['codes']['path']))
            for row in [r for r in p['rows'] if r['sample_id']==sid]:
                guard(run)
                start=time.monotonic();forced=np.load(row['codes']['path']);directory=out/row['id'];directory.mkdir()
                save(directory/'started.json',{'pid':os.getpid(),'unix':time.time(),'id':row['id']})
                # No acoustic decode needed for probability bank; original decode
                # is restored before continuing. This occurs strictly after logits.
                decode=w.model.speech_tokenizer.decode
                def dummy_decode(encoded):return [np.zeros(24000,dtype=np.float32)],24000
                w.model.speech_tokenizer.decode=dummy_decode
                seed(row['seed']);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('TF unit900s')));signal.alarm(900)
                try:
                    kwargs=dict(p['generation_kwargs']);kwargs['max_new_tokens']=len(forced)+1
                    with torch.inference_mode(),h.capture_official(w) as cap,trace_full(w,forced) as tr:
                        w.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**kwargs)
                    assert np.array_equal(cap['new_codes'][0],forced),'full-history resampling'
                    expected=[(f,cb) for f in range(len(forced)) for cb in range(16)]+[(len(forced),0)]
                    assert [(r['frame'],r['cb']) for r in tr['rows']]==expected
                    persist_trace(directory/'trace.npz',tr)
                    result={'status':'COMPLETE','id':row['id'],'reader':name,'content_frames':len(forced),'elapsed_s':time.monotonic()-start,'forced_codes_exact':True,'trace_rows':len(tr['rows']),'qualified':row['qualified'],'trace_sha':sha(directory/'trace.npz')}
                    save(directory/'result.json',result);completed.append(result)
                    print(json.dumps(result),flush=True)
                finally:w.model.speech_tokenizer.decode=decode;signal.alarm(0)
        save(out/'completion.json',{'status':'COMPLETE','count':len(completed),'rows':completed})
    except BaseException as e:save(out/'failure.json',{'status':'FAILED','completed':len(completed),'error':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);ap.add_argument('--model');ap.add_argument('--run',required=True);a=ap.parse_args()
    if a.stage=='prepare':prepare(Path(a.run))
    else:run_bank(a.model,Path(a.run))
