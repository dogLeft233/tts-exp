"""Shared isolated runtime for within-checkpoint normalization controls."""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import ast,argparse,json,hashlib,os,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
OFFICIAL=ROOT/'official'
DATA=ROOT/'data'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
 tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
def tensor_sha(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):
  if not torch.is_tensor(v):continue
  v=v.detach().cpu().contiguous();h.update(k.encode()+b'\0'+str(v.dtype).encode()+str(tuple(v.shape)).encode()+v.numpy().tobytes())
 return h.hexdigest()
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def set_rng(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
@contextmanager
def preserve_rng():
 r=rng_state()
 try:yield
 finally:set_rng(r)
def seed(step,domain):
 value=int.from_bytes(hashlib.sha256(f'42/{step}/{domain}'.encode()).digest()[:4],'little')
 random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def configure():
 os.chdir(OFFICIAL);torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def protocol():return json.loads((ROOT/'protocol.json').read_text())
def parser():
 tree=ast.parse((OFFICIAL/'main.py').read_text());block=next(n for n in tree.body if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id=='__name__' for x in ast.walk(n.test)))
 nodes=[]
 for n in block.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='opt' for t in n.targets):break
  nodes.append(n)
 scope={'argparse':argparse};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OFFICIAL/'main.py'),'exec'),scope);return scope['parser']
def feature(mid,split,norm,gain):
 receipt=json.loads((ROOT/'features'/'receipt.json').read_text());assert receipt['status']=='COMPLETED'
 r=next(r for r in receipt['conditions'] if r['split']==split and r['arm_id']==(mid if split=='train' else 'clean') and r['normalize']==norm and r['input_scale']==gain)
 path=ROOT/'features'/r['file'];assert sha(path)==r['sha256'];return path,r
def dataset(mid,split,norm,gain=1.0):
 assert norm is True and gain in [0.5,1.0]
 from nerf_triplane.provider import NeRFDataset
 path,row=feature(mid,split,norm,gain);folder=ROOT/'datasets'/(mid if split=='train' else 'clean')/split/('on' if norm else 'off')/f's{gain:.2f}'
 opt=parser().parse_args([str(folder),'-O','--asr_model','hubert','--preload','0','--seed','42','--ckpt','scratch','--iters','60000','--num_rays','4096'])
 opt.fp16=False;opt.exp_eye=True;opt.cuda_ray=True
 assert not opt.portrait and not opt.finetune_lips and not opt.torso and not opt.emb and opt.patch_size==1
 assert not opt.test_train and not opt.train_camera and opt.aud==''
 d=NeRFDataset(opt,device=torch.device('cuda'),type=split)
 frames=json.loads((folder/f'transforms_{split}.json').read_text())['frames'];ids=[f['aud_id'] for f in frames]
 raw=np.load(path,allow_pickle=False);assert max(ids)<len(raw)
 expected=raw[ids].transpose(0,2,1)
 assert len(d.poses)==len(d.images)==len(d.auds)==len(frames)=={'train':4150,'val':550,'test':1100}[split]
 assert np.array_equal(d.auds.cpu().numpy(),expected)
 d.source_frames=frames;return d
def network(opt):
 from nerf_triplane.network import NeRFNetwork,AudioNet
 model=NeRFNetwork(opt)
 # Both paired conditions use exactly this architecture and complete initialization.
 model.audio_in_dim=768;model.audio_net=AudioNet(dim_in=768,dim_aud=model.audio_dim)
 return model.cuda()
def bind(model,d):model.aud_features=d.auds;model.eye_area=d.eye_area;model.poses=d.poses
def metadata(model):return {k:getattr(model,k) for k in ['mean_count','mean_density','iter_density','local_step'] if hasattr(model,k)}
def restore_metadata(model,m):
 for k,v in m.items():setattr(model,k,v)
def finite(model,grad=False):
 vals=[v for v in model.parameters() if v.is_floating_point()]
 if grad:vals=[v.grad for v in vals if v.grad is not None]
 assert vals and bool(torch.stack([torch.isfinite(v).all() for v in vals]).all()),'nonfinite gradient' if grad else 'nonfinite model'
def density_finite(model):
 assert torch.isfinite(model.density_grid).all() and np.isfinite(float(model.mean_density)),'nonfinite density refresh'
def metrics(pred,truth,source):
 error=pred.float()-truth.float();lms=np.loadtxt(DATA/f'source/May/ori_imgs/{source}.lms')
 x0,y0=np.maximum(np.floor(lms[48:68].min(0)).astype(int)-12,0);x1,y1=np.minimum(np.ceil(lms[48:68].max(0)).astype(int)+13,512)
 mouth=error[:,y0:y1,x0:x1]
 return dict(MSE=float((error**2).mean()),L1=float(error.abs().mean()),mouth_bbox_MSE=float((mouth**2).mean()),mouth_bbox_L1=float(mouth.abs().mean()))
def validation(model,ema,sets,rows):
 from nerf_triplane.utils import Trainer
 results=[]
 with preserve_rng(),ema.average_parameters(),torch.inference_mode():
  model.eval()
  for norm,d in sets.items():
   owner=SimpleNamespace(model=model,opt=d.opt,criterion=torch.nn.L1Loss(reduction='none'))
   per=[]
   for idx in rows:
    y=Trainer.eval_step(owner,d.collate([idx]));assert torch.isfinite(y[0]).all()
    per.append(dict(index=idx,source_id=d.source_frames[idx]['img_id'],**metrics(y[0],y[5],d.source_frames[idx]['img_id'])))
   results.append(dict(normalize=norm,per_frame=per,mean={k:float(np.mean([r[k] for r in per])) for k in ['MSE','L1','mouth_bbox_MSE','mouth_bbox_L1']}))
 model.train();return results
