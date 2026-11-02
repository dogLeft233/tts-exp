"""Paired input-normalization ablation, same report52 weights/PCM/clock/support.

No training, no internal network norm changes. Creates fresh scientific outputs.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from pathlib import Path
import sys, json, time, subprocess, shutil, traceback, gc
OLD = Path('F:/Datasets/tts-exp/encoder_survey_20261018')
OUT = Path('F:/Datasets/tts-exp/encoder_normalization_ablation_20261019')
sys.path.insert(0, str(OLD/'deps'))
import numpy as np
import requests
from prepare import sha, digest
from run import unit, centers, write

IDS = ['hubert_base','hubert_chinese','wav2vec2_base','data2vec_base',
       'hubert_large','xlsr53','wavlm_large','w2vbert2']
ARMS = ['natural_fixed_float32','natural_plus3db','natural_minus3db']

def frozen_protocol():
    p = json.loads((OLD/'protocol.json').read_text(encoding='utf8'))
    oldresults = json.loads((OLD/'analysis/results.json').read_text(encoding='utf8'))
    selected = [s for mid in IDS for s in p['models'] if s['id']==mid]
    for s in selected:
        s['source'] = json.loads((OLD/'models'/s['id']/'model_source.json').read_text(encoding='utf8'))
        s['old_receipt_sha256'] = sha(OLD/'models'/s['id']/'receipt.json')
    return dict(status='FROZEN_BEFORE_ABLATION_FORWARD', models=selected, arms=ARMS,
        old_protocol_sha256=sha(OLD/'protocol.json'), old_results_sha256=sha(OLD/'analysis/results.json'),
        rows=p['rows'], coverage=oldresults['coverage'],
        intervention='Only do_normalize=False for seven waveform processors; do_normalize_per_mel_bins=False for W2vBERT. All learned/internal norms untouched.',
        design='Same report52 CAL20 10 exposed male speakers/20 sentences, same exact PCM/plus3/minus3, no new TTS or video/SyncNet/training.',
        layers='Save 0,1,2,middle,last; summarize all hidden states. Original 663/516 full/inner shared phone IDs frozen, no new support selection.',
        controls='Same pinned weight/config bodies by SHA; first clip normalized 3-arm replay vs saved report52 last frames (maxabs<=1e-5, relL2<=1e-6), off fixed repeat; same output clocks/masks; finite arrays.',
        signal_gate='Same original RMS>=1e-5 and >10*repeat direction floor. Unit frame then mean phone then unit. Sentence/speaker equal weight.',
        limits='Input normalization ablation may shift input outside pretraining distribution; does not establish semantic quality or TFG/Sync-C improvement.',
        resource='Local RTX4060 FP32, TF32 off, batch1, allocator <=50%; Ubuntu training untouched; one checkpoint cache at a time; retain all science.',
        producer_sha256=sha(__file__))

def download(s):
    folder=OUT/'weights'/s['id']; folder.mkdir(parents=True,exist_ok=True)
    for r in s['source']['files']:
        dest=folder/r['file']
        if dest.exists() and sha(dest)==r['sha256']: continue
        if shutil.disk_usage(folder).free < r['bytes']+650*1024**2: raise RuntimeError('weight space reserve')
        url='https://huggingface.co/'+s['repo']+'/resolve/'+s['source']['revision']+'/'+r['file']
        tmp=dest.with_suffix(dest.suffix+'.partial')
        for attempt in range(4):
            try:
                with requests.get(url,stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    with tmp.open('wb') as f:
                        for b in response.iter_content(4*1024**2): f.write(b)
                if tmp.stat().st_size!=r['bytes'] or sha(tmp)!=r['sha256']: raise ValueError('pinned body mismatch')
                os.replace(tmp,dest); break
            except Exception:
                if attempt==3: raise
                time.sleep(3)
        print('DOWNLOADED',s['id'],r['file'],r['bytes'],flush=True)
    return folder

def worker(mid):
    import torch, transformers
    from transformers import AutoModel,AutoFeatureExtractor
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    s=next(x for x in p['models'] if x['id']==mid)
    dest=OUT/'models'/mid; dest.mkdir(parents=True,exist_ok=False)
    r=dict(id=mid,status='RUNNING',pid=os.getpid(),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),torch=torch.__version__,
        transformers=transformers.__version__,numpy=np.__version__,forward_count=0,clips=[])
    write(dest/'receipt.json',r)
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(20261018)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.cuda.set_per_process_memory_fraction(.5)
        folder=download(s)
        processor=AutoFeatureExtractor.from_pretrained(folder,local_files_only=True)
        r['original_processor']=processor.to_dict()
        if s['kind']=='wave': assert processor.do_normalize is True
        model,load=AutoModel.from_pretrained(folder,local_files_only=True,attn_implementation='eager',
            output_loading_info=True,low_cpu_mem_usage=True)
        assert not load['missing_keys'] and not load['mismatched_keys']
        r['loading_info']=load;model=model.eval().to('cuda')
        count=0
        def encode(x,normalized):
            nonlocal count
            if s['kind']=='wave':
                processor.do_normalize=normalized
                b=processor(x,sampling_rate=16000,return_tensors='pt')
                b={k:v for k,v in b.items() if k!='attention_mask'}
            else:
                b=processor(x,sampling_rate=16000,return_tensors='pt',do_normalize_per_mel_bins=normalized)
            used={k:v.cpu().numpy() for k,v in b.items()}
            with torch.inference_mode():
                y=model(**{k:v.to('cuda') for k,v in b.items()},output_hidden_states=True,return_dict=True)
            count+=1;r['forward_count']=count
            arr=[z[0].cpu().numpy().astype('float32') for z in y.hidden_states]
            ts=centers(s,model.config,len(arr[-1]));mask=ts<len(x)/16000
            if s['kind']=='fbank': mask &= used['attention_mask'][0].astype(bool)
            arr=[z[mask] for z in arr];ts=ts[mask]
            assert all(np.isfinite(z).all() and np.min(np.linalg.norm(z.astype(float),axis=1))>1e-12 for z in arr)
            return arr,ts,used
        first=p['rows'][0]; replay=[]
        with np.load(first['input'],allow_pickle=False) as xs, np.load(OLD/'models'/mid/(first['id']+'.npz'),allow_pickle=False) as z:
            for ci,arm in enumerate(ARMS):
                a,ts,_=encode(xs[arm],True);oldci=[2,4,5][ci]
                old=z['last_frames_'+str(oldci)];assert np.array_equal(ts,z['times_'+str(oldci)])
                err=float(np.max(abs(a[-1]-old)));rel=float(np.linalg.norm(a[-1].astype(float)-old)/np.linalg.norm(old.astype(float)))
                replay.append(dict(arm=arm,maxabs=err,relative_l2=rel))
                assert err<=1e-5 and rel<=1e-6, replay
        r['normalized_replay']=replay;write(dest/'receipt.json',r)
        stats=[]
        for row in p['rows']:
            assert sha(row['input'])==row['sha256']
            saved={};seq=[];useds=[];conds=[]
            with np.load(row['input'],allow_pickle=False) as inp:
                for ci,arm in enumerate(ARMS):
                    x=inp[arm];assert digest(x)==row['pcm_sha256'][arm]
                    a,ts,used=encode(x,False)
                    if ci: assert np.array_equal(ts,seq[0][1])
                    if row==first and ci==0:
                        ar,tr,ur=encode(x,False);assert np.array_equal(ts,tr)
                        repeats=[float(np.sqrt(np.mean(np.sum((unit(z)-unit(zr))**2,axis=1)))) for z,zr in zip(a,ar)]
                        r['repeat_chord_rms']=repeats;r['repeat_maxabs']=[float(np.max(abs(z-zr))) for z,zr in zip(a,ar)]
                        saved['repeat_last']=ar[-1]
                    saved['last_frames_'+str(ci)]=a[-1];saved['times_'+str(ci)]=ts
                    seq.append((a,ts));useds.append(used)
                    conds.append(dict(arm=arm,pcm_sha256=digest(x),samples=len(x),
                        rms=float(np.sqrt(np.mean(x.astype(float)**2))),peak=float(abs(x).max()),frames=len(ts),
                        processor={k:dict(shape=list(v.shape),sha256=digest(v)) for k,v in used.items()}))
            nl=len(seq[0][0]);selected=sorted({0,1,2,nl//2,nl-1});r['layer_count']=nl
            # Exact paired timestamps vs original arrays, independent of input normalization.
            oldrec=json.loads((OLD/'models'/mid/'receipt.json').read_text(encoding='utf8'))
            oldclip=next(t for t in oldrec['clips'] if t['id']==row['id'])
            for ci,oldci in enumerate([2,4,5]): assert conds[ci]['frames']==oldclip['conditions'][oldci]['frame_count']
            for domain in ['full_phone','inner50']:
                ids=next(c['common_phone_indices'] for c in p['coverage'][domain] if c['id']==row['id'])
                saved[domain+'_phone_indices']=np.array(ids,dtype='int32')
                phones={ph['index']:ph for ph in row['phones']};inds=[]
                for idx in ids:
                    lo,hi=phones[idx]['natural']
                    if domain=='inner50': lo,hi=lo+.25*(hi-lo),hi-.25*(hi-lo)
                    ix=np.flatnonzero((seq[0][1]>=lo)&(seq[0][1]<hi));assert len(ix);inds.append(ix)
                for li in range(nl):
                    us=[unit(a[li]) for a,_ in seq]
                    dirs=np.stack([np.stack([unit(v[ix].mean(0)) for ix in inds]) for v in us])
                    radii=np.stack([np.array([np.linalg.norm(a[li][ix].astype(float),axis=1).mean() for ix in inds]) for a,_ in seq])
                    if li in selected:
                        saved[f'{domain}_directions_l{li}']=dirs.astype('float32')
                        saved[f'{domain}_radii_l{li}']=radii.astype('float32')
                    for ep,(ca,cb) in {'GAIN_P3':(0,1),'GAIN_M3_SIGNED':(2,0)}.items():
                        local=us[cb]-us[ca]
                        stats.append(dict(id=row['id'],speaker=row['speaker'],domain=domain,layer=li,endpoint=ep,
                            chord_rms=float(np.sqrt(np.mean(np.sum(local**2,axis=1)))),
                            mean_phone_angle_deg=float(np.degrees(np.arccos(np.clip(np.sum(dirs[ca]*dirs[cb],axis=-1),-1,1))).mean()),
                            radius_ratio=float(np.mean(radii[cb]/radii[ca])),
                            delta_mean=(dirs[cb]-dirs[ca]).mean(0).tolist()))
            proc={}
            key='input_values' if s['kind']=='wave' else 'input_features'
            for ci,ep in [(1,'GAIN_P3'),(2,'GAIN_M3')]:
                aa=useds[0][key].astype(float);bb=useds[ci][key].astype(float)
                proc[ep]=float(np.linalg.norm(bb-aa)/np.linalg.norm(aa))
            file=dest/(row['id']+'.npz');np.savez_compressed(file,**saved)
            r['clips'].append(dict(id=row['id'],file=file.name,sha256=sha(file),conditions=conds,processor_relative_l2=proc))
            write(dest/'clip_statistics.json',stats);r['forward_count']=count;write(dest/'receipt.json',r)
            print('ENCODED_OFF',mid,len(r['clips']),'/20',flush=True)
            del seq,useds;gc.collect()
        assert count==64 and len(r['clips'])==20
        r.update(status='COMPLETED',forward_count=count,end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        write(dest/'receipt.json',r)
    except Exception as e:
        r.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());write(dest/'receipt.json',r);raise

def main():
    if len(sys.argv)>1: worker(sys.argv[1]);return
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'protocol.json',frozen_protocol())
    (OUT/'normalization_ablation.py').write_bytes(Path(__file__).read_bytes())
    for name in ['run.py','prepare.py']:
        (OUT/name).write_bytes((Path(__file__).parent/name).read_bytes())
    status=[]
    for mid in IDS:
        with (OUT/(mid+'.log')).open('xb') as f:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),mid],stdout=f,stderr=subprocess.STDOUT)
            write(OUT/'active.json',dict(model=mid,pid=child.pid,launched_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            rc=child.wait()
        status.append(dict(model=mid,rc=rc));write(OUT/'batch_status.json',status)
        print('MODEL_DONE',mid,rc,flush=True)
        if rc!=0: raise RuntimeError('stopped at failed checkpoint '+mid)
        # Only exact fresh public weight cache is disposable. All scientific bodies retained.
        root=(OUT/'weights'/mid).resolve()
        assert root.parent==(OUT/'weights').resolve()
        for file in root.iterdir():
            if file.suffix in ['.bin','.safetensors']:
                assert file.resolve().parent==root;file.unlink()
    write(OUT/'terminal.json',dict(status='COMPLETED',models=len(status),forwards=64*len(status)))

if __name__=='__main__': main()
