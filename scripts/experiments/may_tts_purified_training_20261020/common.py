"""Baseline50 API, adapting only data, isolated paths and expected row count."""
from pathlib import Path
import ast,argparse,hashlib,json,os,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;OFFICIAL=ROOT/'official';DATA=ROOT/'data';FEATURES=ROOT/'features'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def tensor_state_sha(state):
 h=hashlib.sha256()
 for k,x in sorted(state.items()):
  x=x.detach().cpu().contiguous();h.update(k.encode()+b'\0'+str(x.dtype).encode()+str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
 return h.hexdigest()
def seed_domain(seed,step,domain):
 v=int.from_bytes(hashlib.sha256(f'{seed}/{step}/{domain}'.encode()).digest()[:4],'little')
 random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v);return v
def label(db):return {0:'gain_0',-6:'gain_m6',6:'gain_p6'}[db]
def parser():
 tree=ast.parse((OFFICIAL/'main.py').read_text());block=next(n for n in tree.body if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id=='__name__' for x in ast.walk(n.test)))
 nodes=[]
 for n in block.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='opt' for t in n.targets):break
  nodes.append(n)
 scope={'argparse':argparse};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OFFICIAL/'main.py'),'exec'),scope);return scope['parser']
def configure():
 os.chdir(OFFICIAL);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def options(split,db,protocol):
 folder=DATA/'arms'/split/label(db)
 if split=='train':
  assert db==0 and json.loads((FEATURES/'receipt.json').read_text())['passed'];feature=FEATURES/'train_padded.npy'
  q=next(x for x in json.loads((FEATURES/'receipt.json').read_text())['rows'] if x['split']=='train')
 else:
  feature=ROOT/'historical_features'/split/label(db)/'ave_padded.npy'
  q=next(x for x in json.loads((ROOT/'historical_features/provider_qualification.json').read_text())['arms'] if x['split']==split and x['gain_db']==db)
 assert sha(folder/'aud.wav')==q.get('wav_sha256',q.get('audio_sha256'))
 assert sha(feature)==q['padded_sha256']
 opt=parser().parse_args([str(folder),'-O','--asr_model','ave','--preload','0','--seed',str(protocol['seed']),'--ckpt','scratch','--iters',str(protocol['steps_per_model']),'--num_rays',str(protocol['num_rays'])])
 opt.fp16=False;opt.exp_eye=True;opt.cuda_ray=True;opt.verified_ave_cache=str(feature)
 assert not opt.finetune_lips and opt.patch_size==1 and not opt.torso and not opt.train_camera and not opt.test_train and opt.aud==''
 assert opt.max_steps==protocol['max_steps_per_ray']
 return opt
def dataset(split,db,protocol):
 from nerf_triplane.provider import NeRFDataset
 d=NeRFDataset(options(split,db,protocol),device=torch.device('cuda'),type=split)
 rows=json.loads((Path(d.opt.path)/f'transforms_{split}.json').read_text())['frames'];expected={'train':3849,'val':550,'test':1100}[split]
 assert len(rows)==len(d.poses)==len(d.auds)==len(d.images)==expected
 features=np.load(d.opt.verified_ave_cache,allow_pickle=False);ids=np.array([r['aud_id'] for r in rows])
 assert np.array_equal(d.auds.cpu().numpy(),features[ids,None,:]);d.source_frames=rows
 return d
def bind_training_state(model,train):model.aud_features=train.auds;model.eye_area=train.eye_area;model.poses=train.poses
def model_metadata(model):return {n:getattr(model,n) for n in ['mean_count','mean_density','iter_density','local_step'] if hasattr(model,n)}
def load_metadata(model,metadata):
 for k,v in metadata.items():setattr(model,k,v)
