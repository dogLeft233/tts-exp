"""Full May train, fixed report65 noise; all eval drivers have no added noise."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import sys,json,hashlib,traceback
from protocol import OUT,DATA,sha,write
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/deps')
import numpy as np
from contrast import mixed_input
def main():
 import torch,soundfile as sf
 from transformers import AutoModel,AutoFeatureExtractor
 p=json.loads((OUT/'protocol.json').read_text());dest=OUT/'features';dest.mkdir(exist_ok=False)
 parent=Path(p['initialization_parent']);weights=parent/'weights/wav2vec2_base'
 for r in p['models'][0]['source']['files']:assert sha(weights/r['file'])==r['sha256']
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
 model,loading=AutoModel.from_pretrained(weights,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
 assert not loading['missing_keys'] and not loading['mismatched_keys']
 model=model.eval().cuda();model.requires_grad_(False)
 processor=AutoFeatureExtractor.from_pretrained(weights,local_files_only=True)
 receipt=dict(status='RUNNING',conditions=[],forwards=0,precision='FP32',tf32=False,source=p['models'][0])
 def forward(x):
  with torch.inference_mode():z=model(input_values=torch.from_numpy(x[None]).cuda()).last_hidden_state[0].cpu().numpy()
  receipt['forwards']+=1;assert z.dtype==np.float32 and np.isfinite(z).all();return z
 try:
  for split in ['train','val','test']:
   raw,sr=sf.read(DATA/'arms'/split/'gain_0/aud.wav',dtype='float64');assert sr==16000
   x=raw.astype(np.float32);native=(x-x.mean())/np.sqrt(x.var()+1e-7)
   assert np.array_equal(native,processor(x,sampling_rate=16000,return_tensors='np').input_values[0])
   conditions=p['arms'] if split=='train' else [dict(id='clean',kind='none',snr_db=None,input_scale=s) for s in [1.,.5]]
   for arm in conditions:
    noise=None
    if arm['snr_db'] is not None:
     pin=next(r for r in p['noise_pins'] if r['kind']==arm['kind']);assert sha(OUT/pin['path'])==pin['sha256'];noise=np.load(OUT/pin['path'])
    actual=mixed_input(native,noise,arm['snr_db'],arm['input_scale']);parts=[];chunks=[]
    for start in range(0,len(actual),320000):
     piece=actual[start:min(start+320080,len(actual))]
     if len(piece)<400:continue
     z=forward(piece);parts.append(z);chunks.append(dict(start=start,samples=len(piece),input_sha256=hashlib.sha256(piece.tobytes()).hexdigest(),last_sha256=hashlib.sha256(z.tobytes()).hexdigest()))
    hidden=np.concatenate(parts);expected=(len(actual)-80)//320;assert abs(len(hidden)-expected)<=1
    if len(hidden)<expected:hidden=np.pad(hidden,((0,expected-len(hidden)),(0,0)))
    z=hidden[:expected//2*2].reshape(-1,2,768);filename=f'{split}_{arm["id"]}_s{arm["input_scale"]:.2f}.npy';np.save(dest/filename,z)
    baseline_bridge=None
    if arm['kind']=='none':
     old=Path(p['feature_parent'])/'features/wav2vec2_base'/f'{split}_s{arm["input_scale"]:.2f}.npy'
     cached=np.load(old);assert np.array_equal(z,cached),(split,arm['input_scale'],float(abs(z-cached).max()));baseline_bridge=dict(file=str(old),sha256=sha(old),arrays_exact=True)
    row=dict(split=split,arm_id=arm['id'],kind=arm['kind'],snr_db=arm['snr_db'],input_scale=arm['input_scale'],normalize=True,file=filename,sha256=sha(dest/filename),shape=list(z.shape),source_audio_sha256=sha(DATA/'arms'/split/'gain_0/aud.wav'),native_sha256=hashlib.sha256(native.tobytes()).hexdigest(),input_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),chunks=chunks,baseline_bridge=baseline_bridge)
    receipt['conditions'].append(row);write(dest/'receipt.json',receipt);print(split,arm['id'],arm['input_scale'],'calls',receipt['forwards'],flush=True)
  receipt.update(status='COMPLETED',conditions_count=len(receipt['conditions']),training_updates=0);assert len(receipt['conditions'])==22;write(dest/'receipt.json',receipt)
 except BaseException as e:
  receipt.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());write(dest/'receipt.json',receipt);raise
if __name__=='__main__':main()
