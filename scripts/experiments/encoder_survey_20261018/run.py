"""Sequential native speech encoder geometry; immutable per-model outputs."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_DISABLE_XET']='1'
from pathlib import Path
import sys,json,time,hashlib,gc,traceback,shutil,subprocess
BASE=Path('F:/Datasets/tts-exp/encoder_survey_20261018')
sys.path.insert(0,str(BASE/'deps'))
import numpy as np,requests
from prepare import sha,digest

def write(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 for i in range(100):
  try:os.replace(tmp,p);return
  except PermissionError:
   if i==99:raise
   time.sleep(.1)

def unit(x):
 x=np.asarray(x,dtype=np.float64);n=np.linalg.norm(x,axis=-1,keepdims=True)
 if not np.isfinite(x).all() or np.any(n<1e-12):raise ValueError('nonfinite/degenerate encoder feature')
 return x/n

def download(spec):
 folder=BASE/'weights'/spec['id'];folder.mkdir(parents=True,exist_ok=True)
 pin=folder/'download.json'
 if pin.exists():
  info=json.loads(pin.read_text(encoding='utf8'))
  if all((folder/r['file']).exists() and sha(folder/r['file'])==r['sha256'] for r in info['files']):return folder,info
 api=requests.get('https://huggingface.co/api/models/'+spec['repo'],params={'blobs':'true'},timeout=30);api.raise_for_status();j=api.json()
 siblings={s['rfilename']:s for s in j['siblings']};revision=j['sha']
 weight='model.safetensors' if 'model.safetensors' in siblings else 'pytorch_model.bin'
 files=['config.json','preprocessor_config.json',weight]
 if 'preprocessor_config.json' not in siblings:files.remove('preprocessor_config.json')
 rows=[]
 for name in files:
  dest=folder/name; expected=siblings[name].get('lfs',{}).get('sha256')
  expected_bytes=siblings[name].get('lfs',{}).get('size',siblings[name].get('size'))
  if not (dest.exists() and expected and sha(dest)==expected):
   if expected_bytes and shutil.disk_usage(folder).free<expected_bytes+700*1024**2:raise RuntimeError('insufficient local weight reserve')
   url='https://huggingface.co/'+spec['repo']+'/resolve/'+revision+'/'+name
   tmp=dest.with_suffix(dest.suffix+'.partial')
   for attempt in range(3):
    try:
     with requests.get(url,stream=True,timeout=(30,90)) as r:
      r.raise_for_status()
      with tmp.open('wb') as f:
       for b in r.iter_content(4*1024**2):f.write(b)
     if expected_bytes and tmp.stat().st_size!=expected_bytes:raise ValueError('download byte count')
     if expected and sha(tmp)!=expected:raise ValueError('official LFS SHA mismatch')
     os.replace(tmp,dest);break
    except Exception:
     if attempt==2:raise
     time.sleep(2)
  rows.append({'file':name,'bytes':dest.stat().st_size,'sha256':sha(dest),'official_lfs_sha256':expected})
 info={'repo':spec['repo'],'revision':revision,'files':rows,'api_sha256':hashlib.sha256(api.content).hexdigest()}
 write(pin,info);print('DOWNLOADED',spec['id'],sum(x['bytes'] for x in rows),flush=True)
 return folder,info

def centers(spec,config,n):
 if spec['kind']=='whisper':return np.arange(n)*.02
 if spec['kind']=='fbank':return .0175+np.arange(n)*.02
 rf=1;stride=1
 for k,s in zip(config.conv_kernel,config.conv_stride):rf+=(k-1)*stride;stride*=s
 return ((rf-1)/2+np.arange(n)*stride)/16000

def geometry_summary(clip_deltas):
 sp=sorted({r['speaker'] for r in clip_deltas});vec=[]
 for s in sp:vec.append(np.mean([r['delta'] for r in clip_deltas if r['speaker']==s],axis=0))
 vec=np.stack(vec);norms=np.linalg.norm(vec,axis=-1)
 if np.any(norms<1e-12):return {'status':'DEGENERATE_SPEAKER_DELTA','norms':norms.tolist()}
 u=unit(vec);cos=(u@u.T)[np.triu_indices(len(sp),1)]
 loso=[float(unit(vec[i])@unit(np.delete(vec,i,axis=0).mean(0))) for i in range(len(sp))]
 return {'status':'DESCRIPTIVE','speaker_ids':sp,'speaker_deltas':vec.tolist(),
  'pairwise_cos_mean':float(cos.mean()),'pairwise_cos_min':float(cos.min()),
  'loso_cos_mean':float(np.mean(loso)),'loso_cos_per_speaker':loso,
  'mean_delta_norm':float(np.linalg.norm(vec.mean(0)))}

def worker(spec):
 import torch,transformers
 from transformers import AutoModel,AutoFeatureExtractor,WhisperModel,Wav2Vec2FeatureExtractor
 out=BASE/'models'/spec['id'];out.mkdir(parents=True,exist_ok=False)
 p=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
 receipt={'id':spec['id'],'status':'RUNNING','start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  'protocol_sha256':sha(BASE/'protocol.json'),'source_sha256':sha(__file__),'python':sys.version,
  'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__,'clips':[]}
 write(out/'receipt.json',receipt)
 count=0
 try:
  torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(20261018)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.cuda.set_per_process_memory_fraction(.5)
  folder,info=download(spec);write(out/'model_source.json',info)
  if (folder/'preprocessor_config.json').exists():processor=AutoFeatureExtractor.from_pretrained(folder,local_files_only=True)
  else:
   # Missing official processor config is a deployment failure, not permission to invent normalization.
   raise RuntimeError('official preprocessor_config.json absent')
  receipt['preprocessor']=processor.to_dict()
  if spec['kind']=='whisper':
   full,loadinfo=WhisperModel.from_pretrained(folder,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
   model=full.encoder;del full
  else:model,loadinfo=AutoModel.from_pretrained(folder,local_files_only=True,attn_implementation='eager',output_loading_info=True,low_cpu_mem_usage=True)
  receipt['loading_info']=loadinfo
  # Pretraining-head extra keys are permitted; missing or mismatched encoder parameters are not.
  if loadinfo['missing_keys'] or loadinfo['mismatched_keys']:raise RuntimeError('encoder weights not strict')
  model=model.eval().to('cuda');config=model.config
  count=0;repeat_noise=None;layers_meta=None
  def encode(x):
   nonlocal count
   b=processor(x,sampling_rate=16000,return_tensors='pt')
   b={k:v for k,v in b.items() if k!='attention_mask' or spec['kind']=='fbank'}
   if spec['kind']=='whisper':b={k:v for k,v in b.items() if k=='input_features'}
   used={k:v.detach().cpu().numpy() for k,v in b.items()}
   with torch.inference_mode():y=model(**{k:v.to('cuda') for k,v in b.items()},output_hidden_states=True,return_dict=True)
   states=y.hidden_states
   arr=[z[0].detach().cpu().numpy().astype('float32') for z in states]
   count+=1
   receipt['forward_count']=count
   if not all(np.isfinite(a).all() for a in arr):raise ValueError('nonfinite hidden state')
   return arr,used
  all_clip=[]
  for ri,row in enumerate(p['rows']):
   if shutil.disk_usage(BASE).free<450*1024**2:raise RuntimeError('output reserve below450MiB')
   assert sha(row['input'])==row['sha256']
   with np.load(row['input'],allow_pickle=False) as d:xs=[d[k] for k in p['conditions']]
   saved={};runtime=[];clip_layers=[];seqs=[];proc_inputs=[]
   for ci,x in enumerate(xs):
    assert digest(x)==row['pcm_sha256'][p['conditions'][ci]]
    a,used=encode(x);ts=centers(spec,config,len(a[-1]));keep=ts<len(x)/16000
    native_mask=np.ones(len(ts),dtype=bool)
    if spec['kind']=='fbank' and 'attention_mask' in used:
     native_mask=used['attention_mask'][0].astype(bool)
     assert native_mask.shape==keep.shape
     keep=keep & native_mask
    a=[z[keep] for z in a];ts=ts[keep]
    if ri==0 and ci==0:
     ar,ur=encode(x);ar=[z[keep] for z in ar]
     if spec['kind']=='fbank':assert np.array_equal(ur['attention_mask'],used['attention_mask'])
     repeat_noise=[float(np.sqrt(np.mean(np.sum((unit(z)-unit(zr))**2,axis=1)))) for z,zr in zip(a,ar)]
     receipt['repeat_max_abs']=[float(np.max(abs(z-zr))) for z,zr in zip(a,ar)]
     saved['repeat_last']=ar[-1]
     layers_meta=list(range(len(a)))
     receipt['layer_count']=len(a);receipt['hidden_dimension']=a[-1].shape[-1]
    if ri==0:saved['last_frames_'+str(ci)]=a[-1];saved['times_'+str(ci)]=ts
    seqs.append((a,ts));proc_inputs.append(used)
    runtime.append({'condition':p['conditions'][ci],'samples':len(x),'rms':float(np.sqrt(np.mean(x.astype(float)**2))),
     'peak':float(np.max(abs(x))),'frame_count':len(ts),'pcm_sha256':digest(x),
     'native_invalid_frames_excluded':int((~native_mask).sum()),
     'processor':{k:{'shape':list(v.shape),'sha256':digest(v)} for k,v in used.items()}})
   # One support intersection across the six arms within this encoder.
   for inner in [False,True]:
    supports=[]
    for ph in row['phones']:
     inds=[]
     for ci,(_,ts) in enumerate(seqs):
      lo,hi=ph['tts' if ci in [1,3] else 'natural']
      if inner:lo,hi=lo+.25*(hi-lo),hi-.25*(hi-lo)
      inds.append(np.flatnonzero((ts>=lo)&(ts<hi)))
     if all(len(z) for z in inds):supports.append((ph,inds))
    domain='inner50' if inner else 'full_phone'
    saved[domain+'_phone_indices']=np.array([ph['index'] for ph,_ in supports],dtype='int32')
    if not supports:raise RuntimeError('no matched-phone support')
    selected=sorted({0,1,2,len(layers_meta)//2,len(layers_meta)-1})
    for li in layers_meta:
     u=[unit(a[li]) for a,_ in seqs]
     dirs=np.stack([np.stack([unit(uc[inds[ci]].mean(0)) for _,inds in supports]) for ci,uc in enumerate(u)])
     radii=np.stack([np.array([np.linalg.norm(seqs[ci][0][li][inds[ci]].astype(float),axis=1).mean() for _,inds in supports]) for ci in range(6)])
     if li in selected:
      saved[f'{domain}_directions_l{li}']=dirs.astype('float32')
      saved[f'{domain}_radii_l{li}']=radii.astype('float32')
     for ep,(ca,cb) in p['endpoints'].items():
      delta=dirs[cb]-dirs[ca]
      # Exact same-clock local chord for gain, phone-pool chord for distinct TTS instances.
      local=unit(seqs[cb][0][li])-unit(seqs[ca][0][li]) if ep.startswith('GAIN') else delta
      chord=float(np.sqrt(np.mean(np.sum(local**2,axis=1))))
      clip_layers.append({'id':row['id'],'speaker':row['speaker'],'domain':domain,'layer':li,'endpoint':ep,
        'delta':delta.mean(0).tolist(),'mean_phone_angle_deg':float(np.degrees(np.arccos(np.clip(np.sum(dirs[ca]*dirs[cb],axis=1),-1,1))).mean()),
        'chord_rms':chord,'radius_ratio_mean':float(np.mean(radii[cb]/radii[ca])),'phones':len(supports)})
   proc_response={}
   for ep,(ca,cb) in p['endpoints'].items():
    if not ep.startswith('GAIN'):continue
    key='input_values' if 'input_values' in proc_inputs[ca] else 'input_features'
    aa=proc_inputs[ca][key].astype(float);bb=proc_inputs[cb][key].astype(float)
    proc_response[ep]=float(np.linalg.norm(bb-aa)/max(np.linalg.norm(aa),1e-20))
   npz=out/(row['id']+'.npz');np.savez_compressed(npz,**saved)
   all_clip+=clip_layers
   receipt['clips'].append({'id':row['id'],'file':npz.name,'sha256':sha(npz),'conditions':runtime,'processor_relative_l2':proc_response})
   receipt['forward_count']=count;write(out/'receipt.json',receipt)
   write(out/'clip_statistics.json',all_clip)
   print('ENCODED',spec['id'],ri+1,'/20',flush=True)
   del seqs,proc_inputs;gc.collect()
  results=[]
  for domain in ['full_phone','inner50']:
   for li in layers_meta:
    for ep in p['endpoints']:
     rows=[r for r in all_clip if (r['domain'],r['layer'],r['endpoint'])==(domain,li,ep)]
     for r in rows:r['delta']=np.asarray(r['delta'])
     val=geometry_summary(rows);chord=float(np.mean([r['chord_rms'] for r in rows]))
     val.update(domain=domain,layer=li,endpoint=ep,mean_chord_rms=chord,repeat_chord_rms=repeat_noise[li],
      direction_resolved=bool(chord>=1e-5 and chord>10*repeat_noise[li]),
      mean_angle_deg=float(np.mean([r['mean_phone_angle_deg'] for r in rows])),mean_radius_ratio=float(np.mean([r['radius_ratio_mean'] for r in rows])))
     results.append(val)
  write(out/'summary_native_support.json',{'results':results,'repeat_noise':repeat_noise,'limits':'per-encoder support; common cross-encoder support computed separately before comparison'})
  receipt.update(status='COMPLETED',forward_count=count,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  write(out/'receipt.json',receipt)
 except Exception as e:
  receipt.update(status='FAILED',error=repr(e),forward_count=count,traceback=traceback.format_exc());write(out/'receipt.json',receipt);raise

def main():
 p=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
 if len(sys.argv)>1:
  worker(next(r for r in p['models'] if r['id']==sys.argv[1]));return
 status=[]
 for spec in p['models']:
  path=BASE/'models'/spec['id']/'receipt.json'
  if path.exists():
   old=json.loads(path.read_text(encoding='utf8'))
   if old['status']=='COMPLETED':status.append({'id':spec['id'],'rc':0,'reused_completed':True});continue
   status.append({'id':spec['id'],'rc':-1,'failure_retained':True});continue
  log=BASE/(spec['id']+'.log')
  with log.open('xb') as f:r=subprocess.run([sys.executable,'-u',__file__,spec['id']],stdout=f,stderr=subprocess.STDOUT)
  status.append({'id':spec['id'],'rc':r.returncode});write(BASE/'batch_status.json',status)
  # Public download cache is disposable, unlike saved scientific arrays. Enforce exact task root.
  if r.returncode==0:
   weightroot=(BASE/'weights'/spec['id']).resolve()
   for file in weightroot.glob('*'):
    if file.suffix in ['.bin','.safetensors']:
     assert file.resolve().parent==weightroot
     file.unlink()
  print('MODEL_DONE',spec['id'],r.returncode,flush=True)
 write(BASE/'batch_status.json',status)
if __name__=='__main__':main()
