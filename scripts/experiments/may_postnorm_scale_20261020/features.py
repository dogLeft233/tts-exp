"""Frozen encoders: normalize each complete split first, then multiply before encoder."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import sys,json,hashlib,traceback
from protocol import OUT,DATA,SURVEY,sha,write
sys.path.insert(0,str(SURVEY/'deps'))
import numpy as np
from scaling import postnorm_scale

def worker(mid):
    import torch,soundfile as sf
    from transformers import AutoModel,AutoFeatureExtractor
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));parent=Path(p['parent'])
    spec=next(s for s in p['models'] if s['id']==mid);dest=OUT/'features'/mid;dest.mkdir(parents=True,exist_ok=False)
    folder=parent/'weights'/mid
    for row in spec['source']['files']:assert sha(folder/row['file'])==row['sha256']
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.5)
    model,loading=AutoModel.from_pretrained(folder,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys'] and model.config.hidden_size==768
    model=model.eval().cuda();processor=AutoFeatureExtractor.from_pretrained(folder,local_files_only=True);assert processor.do_normalize is True
    receipt=dict(status='RUNNING',encoder=mid,source=spec['source'],protocol_sha256=sha(OUT/'protocol.json'),conditions=[],new_forward_calls=0)
    inherited=json.loads((parent/'features'/mid/'receipt.json').read_text())
    def forward(y):
        with torch.inference_mode():z=model(input_values=torch.from_numpy(y[None]).cuda()).last_hidden_state[0].cpu().numpy()
        receipt['new_forward_calls']+=1;assert z.dtype==np.float32 and np.isfinite(z).all();return z
    try:
        for split,scales in [('train',[1.0,0.5]),('val',[1.0,0.5]),('test',[1.0,0.5,2.0])]:
            audio=DATA/'arms'/split/'gain_0/aud.wav';raw,sr=sf.read(audio,dtype='float64');assert sr==16000 and raw.ndim==1
            x=raw.astype(np.float32);native=processor(x,sampling_rate=sr,return_tensors='np').input_values[0]
            formula=(x-x.mean())/np.sqrt(x.var()+1e-7);assert np.array_equal(native,formula)
            for scale in scales:
                actual=postnorm_scale(native,scale);name=f'{split}_s{scale:.2f}.npy'
                if scale==1.0:
                    old=next(c for c in inherited['conditions'] if c['split']==split and c['normalize'] and c['gain_db']==0)
                    assert old['processor_sha256']==hashlib.sha256(actual.tobytes()).hexdigest()
                    source=parent/'features'/mid/old['file'];assert sha(source)==old['sha256']
                    import shutil
                    shutil.copyfile(source,dest/name);origin='exact_report54_native_cache';z=np.load(dest/name,allow_pickle=False)
                else:
                    n=len(actual);expected_t=(n-80)//320;parts=[]
                    for start in range(0,n,320000):
                        piece=actual[start:min(start+320080,n)]
                        if len(piece)>=400:parts.append(forward(piece))
                    hidden=np.concatenate(parts);assert abs(len(hidden)-expected_t)<=1
                    if len(hidden)<expected_t:hidden=np.pad(hidden,((0,expected_t-len(hidden)),(0,0)))
                    hidden=hidden[:expected_t];z=hidden[:len(hidden)//2*2].reshape(-1,2,768);np.save(dest/name,z);origin='new_postnorm_encoder_forward'
                    if split=='train':
                        a=forward(actual[:16080]);b=forward(actual[:16080]);assert np.array_equal(a,b)
                        np.save(dest/'repeat_train_s0.50.npy',a)
                stats=lambda v:dict(mean=float(v.astype(float).mean()),rms=float(np.sqrt(np.mean(v.astype(float)**2))),peak=float(abs(v).max()))
                receipt['conditions'].append(dict(split=split,normalize=True,input_scale=scale,file=name,sha256=sha(dest/name),shape=list(z.shape),origin=origin,
                    raw_stats=stats(x),native_stats=stats(native),scaled_stats=stats(actual),source_audio_sha256=sha(audio),
                    processor_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),native_processor_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                    transform_order='native_whole_split_zscore_then_float32_scale_then_frozen_encoder',formula_exact=True))
                write(dest/'receipt.json',receipt);print(mid,split,scale,stats(actual),flush=True)
        receipt.update(status='COMPLETED',inherited_conditions=3,new_conditions=4,peak_memory_reserved=torch.cuda.max_memory_reserved());write(dest/'receipt.json',receipt)
    except BaseException as e:
        receipt.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());write(dest/'receipt.json',receipt);raise
if __name__=='__main__':worker(sys.argv[1])
