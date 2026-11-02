"""Frozen HF encoders, whole-split processor, official 20s chunk/pair clock."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import sys,json,time,shutil,traceback
from protocol import OUT,DATA,SURVEY,sha,write
sys.path.insert(0,str(SURVEY/'deps'))
import numpy as np
import requests
def download(s):
 folder=OUT/'weights'/s['id'];folder.mkdir(parents=True,exist_ok=True)
 for row in s['source']['files']:
  target=folder/row['file']
  if target.exists():assert sha(target)==row['sha256'];continue
  if row['file'].endswith('.json'):
   src=SURVEY/'weights'/s['id']/row['file'];assert sha(src)==row['sha256'];shutil.copyfile(src,target);continue
  assert shutil.disk_usage(OUT).free>row['bytes']+1024**3
  endpoint='https://huggingface.co/'+s['source']['repo']+'.git/info/lfs/objects/batch'
  response=requests.post(endpoint,json={'operation':'download','transfers':['basic'],'objects':[{'oid':row['sha256'],'size':row['bytes']}]},headers={'Accept':'application/vnd.git-lfs+json'},timeout=30)
  response.raise_for_status();url=response.json()['objects'][0]['actions']['download']['href']
  tmp=target.with_suffix('.partial')
  with requests.get(url,stream=True,timeout=(30,120)) as response:
   response.raise_for_status()
   with tmp.open('xb') as f:
    for b in response.iter_content(4*1024**2):f.write(b)
  assert tmp.stat().st_size==row['bytes'] and sha(tmp)==row['sha256'];tmp.replace(target)
  write(folder/'transport.json',dict(endpoint=endpoint,bytes=row['bytes'],sha256=sha(target)))
 return folder
def worker(mid):
 import torch,soundfile as sf,transformers
 from transformers import AutoModel,AutoFeatureExtractor
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));spec=next(s for s in p['models'] if s['id']==mid)
 dest=OUT/'features'/mid;dest.mkdir(parents=True,exist_ok=False)
 receipt=dict(status='RUNNING',encoder=mid,protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(__file__),forward_count=0,conditions=[],torch=torch.__version__,transformers=transformers.__version__)
 write(dest/'receipt.json',receipt)
 try:
  folder=download(spec);torch.set_num_threads(2);torch.set_num_interop_threads(2)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.cuda.set_per_process_memory_fraction(.5)
  model,load=AutoModel.from_pretrained(folder,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
  assert not load['missing_keys'] and not load['mismatched_keys'];assert model.config.hidden_size==768
  model=model.eval().cuda();processor=AutoFeatureExtractor.from_pretrained(folder,local_files_only=True);assert processor.do_normalize is True
  receipt.update(loading_info=load,source=spec['source'],processor_original=processor.to_dict(),model_training=False)
  def forward(y):
   with torch.inference_mode():z=model(input_values=torch.from_numpy(y[None]).cuda()).last_hidden_state[0].cpu().numpy()
   receipt['forward_count']+=1
   assert z.dtype==np.float32 and np.isfinite(z).all();return z
  for split in ['train','val','test']:
   x,sr=sf.read(DATA/'arms'/split/'gain_0/aud.wav',dtype='float64');assert sr==16000 and x.ndim==1
   for norm in [True,False]:
    for gain in ([0] if split!='test' else [-6,0,6]):
     processor.do_normalize=norm;gained=(x*10**(gain/20)).astype(np.float32)
     actual=processor(gained,sampling_rate=16000,return_tensors='np').input_values[0]
     expected=(gained-gained.mean())/np.sqrt(gained.var()+1e-7) if norm else gained
     assert np.array_equal(actual,expected)
     n=len(actual);stride=320;kernel=400;chunk=320000;expected_t=(n-(kernel-stride))//stride
     outputs=[]
     for start in range(0,n,chunk):
      piece=actual[start:min(start+chunk+80,n)]
      if len(piece)>=kernel:outputs.append(forward(piece))
     hidden=np.concatenate(outputs);assert abs(len(hidden)-expected_t)<=1
     if len(hidden)<expected_t:hidden=np.pad(hidden,((0,expected_t-len(hidden)),(0,0)))
     hidden=hidden[:expected_t];hidden=hidden[:len(hidden)//2*2].reshape(-1,2,768)
     name=f'{split}_{"on" if norm else "off"}_g{gain:+d}.npy';np.save(dest/name,hidden)
     if split=='train':
      a=forward(actual[:16080]);b=forward(actual[:16080]);assert np.array_equal(a,b)
      np.save(dest/f'repeat_{"on" if norm else "off"}.npy',a)
     receipt['conditions'].append(dict(split=split,normalize=norm,gain_db=gain,file=name,sha256=sha(dest/name),shape=list(hidden.shape),input_pcm_sha256=__import__('hashlib').sha256(gained.tobytes()).hexdigest(),processor_sha256=__import__('hashlib').sha256(actual.tobytes()).hexdigest(),processor_formula_exact=True,source_audio_sha256=sha(DATA/'arms'/split/'gain_0/aud.wav')))
     write(dest/'receipt.json',receipt);print(mid,split,norm,gain,list(hidden.shape),flush=True)
  assert len(receipt['conditions'])==10
  receipt.update(status='COMPLETED',peak_memory_allocated=torch.cuda.max_memory_allocated(),peak_memory_reserved=torch.cuda.max_memory_reserved());write(dest/'receipt.json',receipt)
 except BaseException as e:
  receipt.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());write(dest/'receipt.json',receipt);raise
if __name__=='__main__':worker(sys.argv[1])
